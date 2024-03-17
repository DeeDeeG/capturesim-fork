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

TESTFILE = "pmcap-Heaven.exe-uncapped-240117-083918.csv"
OBS_FPS = 60.0
OBS_FRAMETIME_MS = 1000.0 / OBS_FPS

gametime_ms = 0.0
obstime_ms = 0.0
last_capture_ms = 0.0
last_render_ms = 0.0
last_capture_frame = -1
last_render_frame = -1

class Disp (enum.Enum):
    UNKNOWN = enum.auto()
    IGNORED = enum.auto()
    CAPTURED = enum.auto()
    COMPOSITED = enum.auto()
    COMPOSITED_DUP = enum.auto()


@dataclass
class GameFrame:
    present_frame: int # The game/input frame number, starting from frame 0 and incrementing by 1 per subsequent row (per subsequent frame) down the capture .csv file.
    present_t_ms: float # Loosely speaking: the total amount of game time elapsed during frames in the capture file, starting from the first through to this frame, inclusive, in ms. Should be larger for each successive frame. Technically speaking: The sum of msBetweenPresents columns for each frame in the PresentMon capture .csv file processed so far, starting from the first frame through to this one, inclusive.
    capture_t_ms: Optional[float] = None # Is this useful? # The timestamp of the simulated moment this frame was "captured" in the simulated game capture loop.
    composite_t_ms: Optional[float] = None # Timestamp of when the simulated OBS render loop "composited" this frame's visuals out to the hypothetical OBS final output (e.g. stream or recording, for a real-world analogy).
    composite_frame: Optional[int] = None # This frame's position in the list of composited frames, if this one eventually got composited. Set during the simulated render loop, not ahead of time.

    disposition: Disp = Disp.UNKNOWN  # A sort of "status" of how this frame is being handled so far. Will be updated throughout different stages of the simulation.


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
            yield GameFrame(
                present_frame=rownum,
                present_t_ms=self.gametime_ms,
                disposition=Disp.UNKNOWN,
            )


# FIXME: Right now this just modifies frames in-place where needed, rather
# than returning an updated one. This may or may not be the right interface
class GameCapture:
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

    frame_detail_print("\n\n===== OUTPUT/COMPOSITED FRAMES =====")
    prev_present_frame = 0
    prev_present_time = 0.0
    gaplist_frames = []
    gaplist_times = []

    for frame in obs.composited_framelist:
        frame_gap = frame.present_frame - prev_present_frame
        prev_present_frame = frame.present_frame
        time_gap = frame.present_t_ms - prev_present_time
        prev_present_time = frame.present_t_ms

        gaplist_frames.append(frame_gap)
        gaplist_times.append(time_gap)

        dupstr = " DUP" if frame.disposition == Disp.COMPOSITED_DUP else ""

        frame_detail_print(f"oframe {frame.composite_frame} @ {frame.composite_t_ms:0.3f}ms, pframe {frame.present_frame} @ {frame.present_t_ms:0.3f}ms, gap {frame_gap} frames, {time_gap:0.3f}ms{dupstr}")

    print("\n\n===== STATS =====")
    print(f"Presented frames: {len(presented_framelist)}")
    print(f"Captured frames: {len(captured_framelist)} ({len(captured_framelist) - len(obs.composited_framelist)} unused)")
    print(f"Composited/output frames: {len(obs.composited_framelist)}")

    g_med = statistics.median(gaplist_frames)
    g_min = min(gaplist_frames)
    g_max = max(gaplist_frames)
    g_stddev = statistics.stdev(gaplist_frames)
    print(
        f"\nFrame number gaps: {g_med:0.2f} med, {g_min} min, {g_max} max, {g_stddev:0.2f} stddev")

    g_med = statistics.median(gaplist_times)
    g_min = min(gaplist_times)
    g_max = max(gaplist_times)
    g_stddev = statistics.stdev(gaplist_times)
    print(
        f"Frame time gaps: {g_med:0.3f} med, {g_min:0.3f} min, {g_max:0.3f} max, {g_stddev:0.3f} stddev")

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
