#!/usr/bin/env python
import argparse
import csv
import enum
import io
import os
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, List, Optional, Tuple, Union

from tdvutil.argparse import CheckFile
from xopen import xopen

### v THESE GLOBAL VARS NOT USED OTHER THAN OBS_FPS?
TESTFILE = "pmcap-Heaven.exe-uncapped-240117-083918.csv"
OBS_FPS = 60.0
OBS_FRAMETIME_MS = 1000.0 / OBS_FPS

gametime_ms = 0.0
obstime_ms = 0.0
last_capture_ms = 0.0
last_render_ms = 0.0
last_capture_frame = -1
last_render_frame = -1
### ^ THESE GLOBAL VARS NOT USED OTHER THAN OBS_FPS?

class Disp (enum.Enum):
    UNKNOWN = enum.auto()
    IGNORED = enum.auto()
    CAPTURED = enum.auto()
    COMPOSITED = enum.auto()
    COMPOSITED_DUP = enum.auto()


@dataclass
class GameFrame:
    present_frame: int # The game/input frame number, starting from frame 0 and incrementing by 1 per subsequent row (per subsequent frame) down the capture .csv file.
    front_edge_timestamp_s: float # Timestamp of the "front edge" (beginning/moment when initially called) of this frame's present call.
    back_edge_timestamp_s: float # Is this computed correctly/accurately? # Timestamp of the "back edge" (end/moment of return) of this frame's present call.
    present_t_ms: float # The amount of time that has passed so far in-game, from the beginning of the PresentMon capture to now, as a sum of msBetweenPresents columns for each frame processed so far. Updated throughout the simulation as new frames are processed.
    capture_t_ms: Optional[float] = None # Is this useful? # The timestamp of the simulated moment this frame was "captured" in the simulated game capture loop.
    composite_t_ms: Optional[float] = None # Timestamp of when the simulated OBS render loop "composited" this frame's visuals out to the hypothetical OBS final output (e.g. stream or recording, for a real-world analogy).
    capture_frame: Optional[int] = None # This frame's position in the list of captured frames, if this one eventually got captured. Set during the simulated capture loop, not ahead of time.
    composite_frame: Optional[int] = None # This frame's position in the list of composited frames, if this one eventually got composited. Set during the simulated render loop, not ahead of time.

    disposition: Disp = Disp.UNKNOWN  # A sort of "status" of how this frame is being handled so far. Updates throughout the simulation.


class FrameStream:
    filename: Path
    reader: Optional[csv.DictReader] = None
    gametime_ms: float = 0.0  # FIXME: Can we just keep this state in getframes?

    def __init__(self, filename: Path) -> None:
        self.filename = filename
        # self.frames: List[GameFrame] = []

    def getframes(self) -> Generator[GameFrame, None, None]:
        if self.reader is not None:
            raise RuntimeError(f"already reading frames from {self.filename}")

        fh = xopen(self.filename, 'r')
        self.reader = csv.DictReader(fh, delimiter=',')

        for rownum, row in enumerate(self.reader):
            self.gametime_ms += float(row['msBetweenPresents'])
            self.front_edge_timestamp_s = float(row['TimeInSeconds'])
            self.back_edge_timestamp_s = self.front_edge_timestamp_s + (float(row['msInPresentAPI']) / 1000)
            yield GameFrame(
                present_frame=rownum,
                front_edge_timestamp_s=self.front_edge_timestamp_s,
                back_edge_timestamp_s=self.back_edge_timestamp_s,
                present_t_ms=self.gametime_ms,
                disposition=Disp.UNKNOWN,
            )


# FIXME: Right now this just modifies frames in-place where needed, rather
# than returning an updated one. This may or may not be the right interface
class GameCapture:
    last_capture_framenum: int = -1
    last_capture_frame: int = -1
    last_capture_ms: float = 0.0  # last frame captured
    game_time_ms: float = 0.0  # current game timestamp (last frame seen)
    capture_interval_ms: float

    def __init__(self, interval: float) -> None:
        self.capture_interval_ms = interval

    def capture(self, frame: GameFrame) -> bool:
        elapsed = frame.present_t_ms - self.last_capture_ms

        # Time to capture?
        if elapsed < self.capture_interval_ms:
            frame.disposition = Disp.IGNORED
            return False

        # Time to capture!
        self.last_capture_frame = frame.present_frame
        frame.disposition = Disp.CAPTURED
        frame.capture_t_ms = frame.present_t_ms
        frame.capture_frame = self.last_capture_framenum + 1
        self.last_capture_framenum += 1

        # set the last capture time so we know when to capture next
        #
        # if the time elapsed has been really long, go from now.
        if elapsed > self.capture_interval_ms * 2:
            self.last_capture_ms = frame.present_t_ms
            return True

        # else we're on a normal cadance, backdate the last capture
        # time to make it an even multiple of half the OBS render
        # interval
        self.last_capture_ms += self.capture_interval_ms
        return True


class OBS:
    composite_interval_ms: float
    last_composite_framenum: int = -1
    last_composite_t_ms: float = 0.0
    last_capture_frame: Optional[GameFrame] = None
    composited_framelist: List[GameFrame] = []
    unique_composited_framelist: List[GameFrame] = []

    def __init__(self, fps: float) -> None:
        self.composite_interval_ms = 1000.0 / fps

    def next_composite_time(self) -> float:
        return self.last_composite_t_ms + self.composite_interval_ms

    def composite(self, frame: GameFrame) -> bool:
        if frame.disposition not in [Disp.CAPTURED, Disp.COMPOSITED, Disp.COMPOSITED_DUP]:
            print(
                f"WARNING: composite() called on non-captured frame: {frame.present_frame} @ {frame.present_t_ms} ({frame.disposition})", file=sys.stderr)
            return False

        # We depend on the caller to make sure it's actually *time* to composite.
        # This may or may not be a good idea.
        #
        # Take the provided frame and copy the bits to use as the entry in our
        # composited frame list.
        fakeframe = GameFrame(**frame.__dict__)
        fakeframe.composite_frame = self.last_composite_framenum + 1
        fakeframe.composite_t_ms = self.next_composite_time()
        fakeframe.disposition = Disp.COMPOSITED

        # mark the original frame as composited
        frame.composite_frame = self.last_composite_framenum + 1
        frame.composite_t_ms = self.next_composite_time()

        if self.last_capture_frame is not None and frame.present_frame == self.last_capture_frame.present_frame:
            # duplicate frame, mark it in both the frame passed in, and the
            # frame stored in the composited frame list
            frame.disposition = Disp.COMPOSITED_DUP
            fakeframe.disposition = Disp.COMPOSITED_DUP
        else:
            # new frame, not a dup
            frame.disposition = Disp.COMPOSITED

        self.composited_framelist.append(fakeframe)
        if fakeframe.disposition != Disp.COMPOSITED_DUP:
            self.unique_composited_framelist.append(fakeframe)
        self.last_capture_frame = frame

        # move ourself one composite frame forward
        self.last_composite_framenum += 1
        self.last_composite_t_ms = self.next_composite_time()

        return True

#
# main code
#
def parse_args(args: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate OBS capture & compositing")

    parser.add_argument(
        "--presentmon-file", "--pmf",
        type=Path,
        default=None,
        action=CheckFile(must_exist=True),
        help="use specified PresentMon capture file as pframe source",
    )

    parser.add_argument(
        "--capture-ratio", "--cr",
        type=float,
        default=2,
        help="capture no more than [this ratio] * [OBS FPS] times per second, loosely speaking (set to 0 for no limit)",
    )

    parser.add_argument(
        "--stats-only", "--silent", "-s",
        default=False,
        action="store_true",
        help="print only the statistics, not the presented or captured frame info",
    )

    return parser.parse_args(args)

def main(argv: List[str]) -> int:
    args = parse_args(argv)
    if args.presentmon_file is None:
        print("ERROR: no PresentMon file specified", file=sys.stderr)
        return 1

    presented_framelist: List[GameFrame] = []
    captured_framelist: List[GameFrame] = []
    last_captured: Optional[GameFrame] = None

    obs = OBS(OBS_FPS)
    if args.capture_ratio == 0:
        gc = GameCapture(0)
    else:
        gc = GameCapture(obs.composite_interval_ms / args.capture_ratio)

    print(f"Data from: '{args.presentmon_file}'\nComposite rate {OBS_FPS}fps\n")

    framestream = FrameStream(filename=args.presentmon_file)
    for frame in framestream.getframes():
        # is this frame newer than our next expected compositor time? If so,
        # call the compositor on the frame most recently captured. This
        # simulates having the compositor run on a timer without having to
        # call it for every single game frame just to have it reject most of
        # them
        if last_captured is not None:
            while frame.present_t_ms > obs.next_composite_time():
                obs.composite(last_captured)

        captured = gc.capture(frame)
        if captured:
            last_captured = frame
            captured_framelist.append(frame)

        presented_framelist.append(frame)

    # Don't print frame details in stats-only/silent mode
    def frame_detail_print(*fargs):
        if not args.stats_only:
            print(*fargs)


    # we're done, print some stuff
    if not args.stats_only:
        print("===== PRESENTED FRAMES =====")
        for frame in presented_framelist:
            if frame.disposition == Disp.COMPOSITED:
                dispstr = f"CAPTURED + COMPOSITED @ otime {frame.composite_t_ms:0.3f}ms"
                # composited_framelist.append(frame)
            elif frame.disposition == Disp.COMPOSITED_DUP:
                dispstr = f"CAPTURED + COMPOSITED (DUPS) @ otime {frame.composite_t_ms:0.3f}ms"
            else:
                dispstr = frame.disposition.name
            print(f"pframe {frame.present_frame} @ {frame.present_t_ms:0.3f}ms, {dispstr}")

    presentmon_capture_beginning_offset_s = float(round(presented_framelist[0].back_edge_timestamp_s) * 1000)

    prev_front_edge_present_time = 0.0
    prev_back_edge_present_time = presentmon_capture_beginning_offset_s
    gaplist_present_front_edge_times = []
    gaplist_present_back_edge_times = []
    for frame in presented_framelist:
        front_edge_time_gap = frame.present_t_ms - prev_front_edge_present_time
        prev_front_edge_present_time = frame.present_t_ms

        back_edge_time_gap = (frame.back_edge_timestamp_s * 1000) - prev_back_edge_present_time
        prev_back_edge_present_time = frame.back_edge_timestamp_s * 1000

        gaplist_present_front_edge_times.append(front_edge_time_gap)
        gaplist_present_back_edge_times.append(back_edge_time_gap)


    frame_detail_print("\n\n===== CAPTURED FRAMES =====")
    prev_present_frame = 0
    prev_front_edge_present_time = 0.0
    prev_back_edge_present_time = presentmon_capture_beginning_offset_s
    gaplist_captured_frames = []
    gaplist_captured_front_edge_times = []
    gaplist_captured_back_edge_times = []

    for frame in captured_framelist:
        frame_gap = frame.present_frame - prev_present_frame
        prev_present_frame = frame.present_frame

        front_edge_time_gap = frame.present_t_ms - prev_front_edge_present_time
        prev_front_edge_present_time = frame.present_t_ms

        back_edge_time_gap = (frame.back_edge_timestamp_s * 1000) - prev_back_edge_present_time
        prev_back_edge_present_time = frame.back_edge_timestamp_s * 1000

        gaplist_captured_frames.append(frame_gap)
        gaplist_captured_front_edge_times.append(front_edge_time_gap)
        gaplist_captured_back_edge_times.append(back_edge_time_gap)

        frame_detail_print(f"cframe {frame.capture_frame}, pframe {frame.present_frame} @ {frame.present_t_ms:0.3f}ms, gap {frame_gap} frames, {front_edge_time_gap:0.3f}ms (front), {back_edge_time_gap:0.3f}ms (back)")


    frame_detail_print("\n\n===== OUTPUT/COMPOSITED FRAMES =====")
    prev_present_frame = 0
    prev_front_edge_present_time = 0.0
    prev_back_edge_present_time = presentmon_capture_beginning_offset_s
    gaplist_output_frames = []
    gaplist_output_front_edge_times = []
    gaplist_output_back_edge_times = []

    for frame in obs.composited_framelist:
        frame_gap = frame.present_frame - prev_present_frame
        prev_present_frame = frame.present_frame

        front_edge_time_gap = frame.present_t_ms - prev_front_edge_present_time
        prev_front_edge_present_time = frame.present_t_ms

        back_edge_time_gap = (frame.back_edge_timestamp_s * 1000) - prev_back_edge_present_time
        prev_back_edge_present_time = frame.back_edge_timestamp_s * 1000

        gaplist_output_frames.append(frame_gap)
        gaplist_output_front_edge_times.append(front_edge_time_gap)
        gaplist_output_back_edge_times.append(back_edge_time_gap)

        dupstr = " DUP" if frame.disposition == Disp.COMPOSITED_DUP else ""

        frame_detail_print(f"oframe {frame.composite_frame} @ {frame.composite_t_ms:0.3f}ms, cframe {frame.capture_frame}, pframe {frame.present_frame} @ {frame.present_t_ms:0.3f}ms, gap {frame_gap} frames, {front_edge_time_gap:0.3f}ms (front), {back_edge_time_gap:0.3f}ms (back){dupstr}")

    composited_frames_count = len(obs.composited_framelist)
    unique_composited_frames_count = len(obs.unique_composited_framelist)
    unique_frame_percentage = (unique_composited_frames_count / composited_frames_count) * 100

    print("\n\n===== STATS =====")
    print(f"Presented frames: {len(presented_framelist)}")
    print(f"Captured frames: {len(captured_framelist)} ({len(captured_framelist) - unique_composited_frames_count} unused)")
    print(f"Composited/output frames: {len(obs.composited_framelist)} ({unique_composited_frames_count} unique ({unique_frame_percentage:0.3f}% unique, {100 - unique_frame_percentage:0.3f}% doubled))")

    avg_fps = len(presented_framelist) / (presented_framelist[-1].front_edge_timestamp_s - presented_framelist[0].front_edge_timestamp_s)
    print(f"\nInput/game average FPS: {avg_fps:0.3f}")

    g_avg = statistics.mean(gaplist_present_front_edge_times)
    g_med = statistics.median(gaplist_present_front_edge_times)
    g_min = min(gaplist_present_front_edge_times)
    g_max = max(gaplist_present_front_edge_times)
    g_stddev = statistics.stdev(gaplist_present_front_edge_times)
    print(
        f"Input/game frame time gaps (front edge): {g_avg:0.3f} avg, {g_med:0.3f} med, {g_min:0.3f} min, {g_max:0.3f} max, {g_stddev:0.3f} stddev")

    g_avg = statistics.mean(gaplist_present_back_edge_times)
    g_med = statistics.median(gaplist_present_back_edge_times)
    g_min = min(gaplist_present_back_edge_times)
    g_max = max(gaplist_present_back_edge_times)
    g_stddev = statistics.stdev(gaplist_present_back_edge_times)
    print(
        f"Input/game frame time gaps (back edge): {g_avg:0.3f} avg, {g_med:0.3f} med, {g_min:0.3f} min, {g_max:0.3f} max, {g_stddev:0.3f} stddev")


    g_avg = statistics.mean(gaplist_captured_frames)
    g_med = statistics.median(gaplist_captured_frames)
    g_min = min(gaplist_captured_frames)
    g_max = max(gaplist_captured_frames)
    g_stddev = statistics.stdev(gaplist_captured_frames)
    print(
        f"\nCaptured frame number gaps: {g_avg:0.2f} avg, {g_med:0.2f} med, {g_min} min, {g_max} max, {g_stddev:0.2f} stddev")

    g_avg = statistics.mean(gaplist_captured_front_edge_times)
    g_med = statistics.median(gaplist_captured_front_edge_times)
    g_min = min(gaplist_captured_front_edge_times)
    g_max = max(gaplist_captured_front_edge_times)
    g_stddev = statistics.stdev(gaplist_captured_front_edge_times)
    print(
        f"Captured frame time gaps (front edge): {g_avg:0.3f} avg, {g_med:0.3f} med, {g_min:0.3f} min, {g_max:0.3f} max, {g_stddev:0.3f} stddev")

    g_avg = statistics.mean(gaplist_captured_back_edge_times)
    g_med = statistics.median(gaplist_captured_back_edge_times)
    g_min = min(gaplist_captured_back_edge_times)
    g_max = max(gaplist_captured_back_edge_times)
    g_stddev = statistics.stdev(gaplist_captured_back_edge_times)
    print(
        f"Captured frame time gaps (back edge): {g_avg:0.3f} avg, {g_med:0.3f} med, {g_min:0.3f} min, {g_max:0.3f} max, {g_stddev:0.3f} stddev")


    g_avg = statistics.mean(gaplist_output_frames)
    g_med = statistics.median(gaplist_output_frames)
    g_min = min(gaplist_output_frames)
    g_max = max(gaplist_output_frames)
    g_stddev = statistics.stdev(gaplist_output_frames)
    print(
        f"\nOutput/composited frame number gaps: {g_avg:0.2f} avg, {g_med:0.2f} med, {g_min} min, {g_max} max, {g_stddev:0.2f} stddev")

    g_avg = statistics.mean(gaplist_output_front_edge_times)
    g_med = statistics.median(gaplist_output_front_edge_times)
    g_min = min(gaplist_output_front_edge_times)
    g_max = max(gaplist_output_front_edge_times)
    g_stddev = statistics.stdev(gaplist_output_front_edge_times)
    print(
        f"Output/composited frame time gaps (front edge): {g_avg:0.3f} avg, {g_med:0.3f} med, {g_min:0.3f} min, {g_max:0.3f} max, {g_stddev:0.3f} stddev")

    g_avg = statistics.mean(gaplist_output_back_edge_times)
    g_med = statistics.median(gaplist_output_back_edge_times)
    g_min = min(gaplist_output_back_edge_times)
    g_max = max(gaplist_output_back_edge_times)
    g_stddev = statistics.stdev(gaplist_output_back_edge_times)
    print(
        f"Output/composited frame time gaps (back edge): {g_avg:0.3f} avg, {g_med:0.3f} med, {g_min:0.3f} min, {g_max:0.3f} max, {g_stddev:0.3f} stddev")

    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))

sys.exit()


# static inline bool frame_ready(uint64_t interval)
# {
#     static uint64_t last_time = 0;
#     uint64_t elapsed;
#     uint64_t t;

#     if (!interval) {
#         return true;
#     }

#     t = os_gettime_ns();
#     elapsed = t - last_time;

#     if (elapsed < interval) {
#         return false;
#     }

#     last_time = (elapsed > interval * 2) ? t : last_time + interval;
#     return true;
# }
