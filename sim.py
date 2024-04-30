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
    SEED = enum.auto()


@dataclass
class GameFrame:
    present_frame: int # The game/input frame number, starting from frame 0 and incrementing by 1 per subsequent row (per subsequent frame) down the capture .csv file.
    present_t_ms: float # Loosely speaking: the total amount of game time elapsed during frames in the capture file, starting from the first through to this frame, inclusive, in ms. Should be larger for each successive frame. Technically speaking: The sum of msBetweenPresents columns for each frame in the PresentMon capture .csv file processed so far, starting from the first frame through to this one, inclusive.
    back_edge_present_t_ms: float # Same as the above, but based on "back edge" times. These have to be calculated relative to the front edge times, since they are not provided directly in the PresentMon .csv file, only indirectly via front edge times + the msInPresentAPI column of the .csv file.
    capture_t_ms: Optional[float] = None # Is this useful? # The timestamp of the simulated moment this frame was "captured" in the simulated game capture loop.
    composite_t_ms: Optional[float] = None # Timestamp of when the simulated OBS render loop "composited" this frame's visuals out to the hypothetical OBS final output (e.g. stream or recording, for a real-world analogy).
    capture_frame: Optional[int] = None # This frame's position in the list of captured frames, if this one eventually got captured. Set during the simulated capture loop, not ahead of time.
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
            self.back_edge_gametime_ms = self.gametime_ms + float(row['msInPresentAPI'])
            yield GameFrame(
                present_frame=rownum,
                present_t_ms=self.gametime_ms,
                back_edge_present_t_ms=self.back_edge_gametime_ms,
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
        if frame.disposition not in [Disp.CAPTURED, Disp.COMPOSITED, Disp.COMPOSITED_DUP, Disp.SEED]:
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

        if fakeframe.capture_t_ms != None:
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

    seedframe = GameFrame(
        present_frame=-1,
        capture_frame=-1,
        present_t_ms=None,
        back_edge_present_t_ms=None,
        disposition=Disp.SEED,
    )

    for frame in framestream.getframes():
        # is this frame newer than our next expected compositor time? If so,
        # call the compositor on the frame most recently captured. This
        # simulates having the compositor run on a timer without having to
        # call it for every single game frame just to have it reject most of
        # them
        if frame.present_frame == 0:
            # Fill in composites of a made-up filler "n-1" pframe (a "seed frame"),
            # representing the frame just before the capture .csv started,
            # to catch simulated OBS composite times up to when the first frame in the actual capture .csv should be composited.
            # We must ignore these filler frames for stats purposes, but this "seeds" (iterates forward)
            # OBS time so that the composite_t_ms is accurate/reasonable for the first
            # actual presented frames. Specifically, this is necessary if the first frame in the .csv
            # has a present_t_ms slower than the first OBS composite interval.
            # Otherwise we composite this first real, slow pframe too many times, and it distorts stats slightly.
            while obs.next_composite_time() < frame.present_t_ms:
                obs.composite(seedframe)
                last_captured = None

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

    prev_front_edge_present_time = 0.0
    prev_back_edge_present_time = None
    prev_front_edge_time_gap = None
    prev_back_edge_time_gap = None
    gaplist_present_front_edge_times = []
    gaplist_present_back_edge_times = []
    deviationslist_rel_present_front_edge = []
    deviationslist_rel_present_back_edge = []
    deviationslist_abs_present_front_edge = []
    deviationslist_abs_present_back_edge = []
    for frame in presented_framelist:
        front_edge_time_gap = frame.present_t_ms - prev_front_edge_present_time
        prev_front_edge_present_time = frame.present_t_ms

        if prev_back_edge_present_time is None:
            back_edge_time_gap = None
        else:
            back_edge_time_gap = frame.back_edge_present_t_ms - prev_back_edge_present_time
        prev_back_edge_present_time = frame.back_edge_present_t_ms

        if prev_back_edge_time_gap is None:
            front_edge_deviation = None
            back_edge_deviation = None
        else:
            front_edge_deviation = front_edge_time_gap - prev_front_edge_time_gap
            back_edge_deviation = back_edge_time_gap - prev_back_edge_time_gap
            deviationslist_rel_present_front_edge.append(front_edge_deviation)
            deviationslist_rel_present_back_edge.append(back_edge_deviation)
            deviationslist_abs_present_front_edge.append(abs(front_edge_deviation))
            deviationslist_abs_present_back_edge.append(abs(back_edge_deviation))

        gaplist_present_front_edge_times.append(front_edge_time_gap)
        gaplist_present_back_edge_times.append(back_edge_time_gap)

        prev_front_edge_time_gap = front_edge_time_gap
        prev_back_edge_time_gap = back_edge_time_gap

    frame_detail_print("\n\n===== CAPTURED FRAMES =====")
    prev_present_frame = None
    prev_front_edge_present_time = None
    prev_back_edge_present_time = None
    prev_front_edge_time_gap = None
    prev_back_edge_time_gap = None
    gaplist_captured_frames = []
    gaplist_captured_front_edge_times = []
    gaplist_captured_back_edge_times = []
    deviationslist_rel_captured_front_edge = []
    deviationslist_rel_captured_back_edge = []
    deviationslist_abs_captured_front_edge = []
    deviationslist_abs_captured_back_edge = []

    for frame in captured_framelist:
        if prev_present_frame is None:
            # First captured frame has no real gap to report.
            # (It would be the gap between first frame and... nothing? "Undefined"?)
            # So, we won't calculate a gap from the first frame to "nothing" for stats purposes.
            frame_detail_print(f"cframe {frame.capture_frame}, pframe {frame.present_frame} @ {frame.present_t_ms:0.3f}ms, gap N/A")
        else:
            frame_gap = frame.present_frame - prev_present_frame
            front_edge_time_gap = frame.present_t_ms - prev_front_edge_present_time
            back_edge_time_gap = frame.back_edge_present_t_ms - prev_back_edge_present_time
            gaplist_captured_frames.append(frame_gap)
            gaplist_captured_front_edge_times.append(front_edge_time_gap)
            gaplist_captured_back_edge_times.append(back_edge_time_gap)

            skipstr = " SKIP" if frame.present_frame - prev_present_frame > 1 else ""
            gapstr = f"gap {frame_gap} pframes, {front_edge_time_gap:0.3f}ms (front), {back_edge_time_gap:0.3f}ms (back)"
            frame_detail_print(f"cframe {frame.capture_frame}, pframe {frame.present_frame} @ {frame.present_t_ms:0.3f}ms, {gapstr}{skipstr}")

        if prev_back_edge_time_gap is None:
            front_edge_deviation = None
            back_edge_deviation = None
        else:
            front_edge_deviation = front_edge_time_gap - prev_front_edge_time_gap
            back_edge_deviation = back_edge_time_gap - prev_back_edge_time_gap
            deviationslist_rel_captured_front_edge.append(front_edge_deviation)
            deviationslist_rel_captured_back_edge.append(back_edge_deviation)
            deviationslist_abs_captured_front_edge.append(abs(front_edge_deviation))
            deviationslist_abs_captured_back_edge.append(abs(back_edge_deviation))

        # Always update "previous_..." variables, for the next frame to use,
        # regardless of whether we calculated gap stats for *this* frame.
        prev_present_frame = frame.present_frame
        prev_front_edge_present_time = frame.present_t_ms
        prev_back_edge_present_time = frame.back_edge_present_t_ms
        prev_front_edge_time_gap = front_edge_time_gap
        prev_back_edge_time_gap = back_edge_time_gap

    frame_detail_print("\n\n===== OUTPUT/COMPOSITED FRAMES =====")
    prev_present_frame = None
    prev_capture_frame = None
    prev_front_edge_present_time = None
    prev_back_edge_present_time = None
    prev_front_edge_time_gap = None
    prev_back_edge_time_gap = None
    prev_otime_vs_ptime_offset_front_edge = None
    prev_otime_vs_ptime_offset_back_edge = None
    gaplist_output_frames = []
    gaplist_output_front_edge_times = []
    gaplist_output_back_edge_times = []
    deviationslist_rel_output_front_edge = []
    deviationslist_rel_output_back_edge = []
    deviationslist_abs_output_front_edge = []
    deviationslist_abs_output_back_edge = []
    offsetslist_otime_vs_ptime_front_edge = []
    offsetslist_otime_vs_ptime_back_edge = []
    deviationslist_rel_otime_vs_ptime_front_edge = []
    deviationslist_rel_otime_vs_ptime_back_edge = []
    deviationslist_abs_otime_vs_ptime_front_edge = []
    deviationslist_abs_otime_vs_ptime_back_edge = []

    for frame in obs.composited_framelist:
        if frame.capture_t_ms is None:
            # A seed frame that somehow slipped through the cracks?
            print("Warning: Seed frame (fake filler frame) encountered during calculation of output/composited frame stats. Skipping this frame in the gap stats and verbose frame output. Composited frame counts and % unique will be somewhat off.")
            continue

        dupstr = " DUP" if frame.disposition == Disp.COMPOSITED_DUP else ""

        if prev_present_frame is None:
            # First composited frame has no real gap to report.
            # (It would be the gap between first frame and... nothing? "Undefined"?)
            # So, we won't calculate a gap from the first frame to "nothing" for stats purposes.
            frame_detail_print(f"oframe {frame.composite_frame} @ {frame.composite_t_ms:0.3f}ms, cframe {frame.capture_frame}, pframe {frame.present_frame} @ {frame.present_t_ms:0.3f}ms, gap N/A{dupstr}")
        else:
            frame_gap = frame.present_frame - prev_present_frame
            front_edge_time_gap = frame.present_t_ms - prev_front_edge_present_time
            back_edge_time_gap = frame.back_edge_present_t_ms - prev_back_edge_present_time
            gaplist_output_frames.append(frame_gap)
            gaplist_output_front_edge_times.append(front_edge_time_gap)
            gaplist_output_back_edge_times.append(back_edge_time_gap)

            skipstr = " SKIP" if frame.capture_frame - prev_capture_frame > 1 else ""
            gapstr = f"gap {frame_gap} pframes, {front_edge_time_gap:0.3f}ms (front), {back_edge_time_gap:0.3f}ms (back)"
            frame_detail_print(f"oframe {frame.composite_frame} @ {frame.composite_t_ms:0.3f}ms, cframe {frame.capture_frame}, pframe {frame.present_frame} @ {frame.present_t_ms:0.3f}ms, {gapstr}{dupstr}{skipstr}")

        otime_vs_ptime_offset_front_edge = frame.composite_t_ms - frame.present_t_ms
        otime_vs_ptime_offset_back_edge = frame.composite_t_ms - frame.back_edge_present_t_ms
        offsetslist_otime_vs_ptime_front_edge.append(otime_vs_ptime_offset_front_edge)
        offsetslist_otime_vs_ptime_back_edge.append(otime_vs_ptime_offset_back_edge)

        if prev_back_edge_time_gap is None:
            front_edge_deviation = None
            back_edge_deviation = None
        else:
            front_edge_deviation = front_edge_time_gap - prev_front_edge_time_gap
            back_edge_deviation = back_edge_time_gap - prev_back_edge_time_gap
            deviationslist_rel_output_front_edge.append(front_edge_deviation)
            deviationslist_rel_output_back_edge.append(back_edge_deviation)
            deviationslist_abs_output_front_edge.append(abs(front_edge_deviation))
            deviationslist_abs_output_back_edge.append(abs(back_edge_deviation))

            otime_vs_ptime_deviation_front_edge = otime_vs_ptime_offset_front_edge - prev_otime_vs_ptime_offset_front_edge
            otime_vs_ptime_deviation_back_edge = otime_vs_ptime_offset_back_edge - prev_otime_vs_ptime_offset_back_edge
            deviationslist_rel_otime_vs_ptime_front_edge.append(otime_vs_ptime_deviation_front_edge)
            deviationslist_rel_otime_vs_ptime_back_edge.append(otime_vs_ptime_deviation_back_edge)
            deviationslist_abs_otime_vs_ptime_front_edge.append(abs(otime_vs_ptime_deviation_front_edge))
            deviationslist_abs_otime_vs_ptime_back_edge.append(abs(otime_vs_ptime_deviation_back_edge))

        # Always update "previous_..." variables, for the next frame to use,
        # regardless of whether we calculated gap stats for *this* frame.
        prev_present_frame = frame.present_frame
        prev_capture_frame = frame.capture_frame
        prev_front_edge_present_time = frame.present_t_ms
        prev_back_edge_present_time = frame.back_edge_present_t_ms
        prev_front_edge_time_gap = front_edge_time_gap
        prev_back_edge_time_gap = back_edge_time_gap
        prev_otime_vs_ptime_offset_front_edge = otime_vs_ptime_offset_front_edge
        prev_otime_vs_ptime_offset_back_edge = otime_vs_ptime_offset_back_edge

    frame_detail_print("\n\n===== UNIQUE OUTPUT/COMPOSITED FRAMES =====")
    prev_present_frame = None
    prev_capture_frame = None
    prev_composite_frame = None
    prev_composite_time = None
    prev_front_edge_present_time = None
    prev_back_edge_present_time = None
    prev_front_edge_time_gap = None
    prev_back_edge_time_gap = None
    prev_otime_vs_ptime_offset_front_edge = None
    prev_otime_vs_ptime_offset_back_edge = None
    gaplist_unique_output_frames = []
    gaplist_unique_output_frame_oframes = []
    gaplist_unique_output_composite_times = []
    gaplist_unique_output_front_edge_times = []
    gaplist_unique_output_back_edge_times = []
    deviationslist_rel_unique_output_front_edge = []
    deviationslist_rel_unique_output_back_edge = []
    deviationslist_abs_unique_output_front_edge = []
    deviationslist_abs_unique_output_back_edge = []
    offsetslist_unique_otime_vs_ptime_front_edge = []
    offsetslist_unique_otime_vs_ptime_back_edge = []
    deviationslist_rel_unique_otime_vs_ptime_front_edge = []
    deviationslist_rel_unique_otime_vs_ptime_back_edge = []
    deviationslist_abs_unique_otime_vs_ptime_front_edge = []
    deviationslist_abs_unique_otime_vs_ptime_back_edge = []

    for frame in obs.unique_composited_framelist:
        if frame.capture_t_ms is None:
            # A seed frame that somehow slipped through the cracks?
            print("Warning: Seed frame (fake filler frame) encountered during calculation of output/composited frame stats. Skipping this frame in the gap stats and verbose frame output. Composited frame counts and % unique will be somewhat off.")
            continue

        dupstr = " DUP" if frame.disposition == Disp.COMPOSITED_DUP else ""

        if prev_present_frame is None:
            # First composited frame has no real gap to report.
            # (It would be the gap between first frame and... nothing? "Undefined"?)
            # So, we won't calculate a gap from the first frame to "nothing" for stats purposes.
            frame_detail_print(f"oframe {frame.composite_frame} @ {frame.composite_t_ms:0.3f}ms, cframe {frame.capture_frame}, pframe {frame.present_frame} @ {frame.present_t_ms:0.3f}ms, gap N/A{dupstr}")
        else:
            frame_gap = frame.present_frame - prev_present_frame
            composite_frame_gap = frame.composite_frame - prev_composite_frame
            composite_time_gap = frame.composite_t_ms - prev_composite_time
            front_edge_time_gap = frame.present_t_ms - prev_front_edge_present_time
            back_edge_time_gap = frame.back_edge_present_t_ms - prev_back_edge_present_time
            gaplist_unique_output_frames.append(frame_gap)
            gaplist_unique_output_frame_oframes.append(composite_frame_gap)
            gaplist_unique_output_composite_times.append(composite_time_gap)
            gaplist_unique_output_front_edge_times.append(front_edge_time_gap)
            gaplist_unique_output_back_edge_times.append(back_edge_time_gap)

            skipstr = " SKIP" if frame.capture_frame - prev_capture_frame > 1 else ""
            gapstr = f"gap {frame_gap} pframes, {composite_frame_gap} oframes, {composite_time_gap:0.3f}ms otime, {front_edge_time_gap:0.3f}ms (front), {back_edge_time_gap:0.3f}ms (back)"
            frame_detail_print(f"oframe {frame.composite_frame} @ {frame.composite_t_ms:0.3f}ms, cframe {frame.capture_frame}, pframe {frame.present_frame} @ {frame.present_t_ms:0.3f}ms, {gapstr}{dupstr}{skipstr}")

        otime_vs_ptime_offset_front_edge = frame.composite_t_ms - frame.present_t_ms
        otime_vs_ptime_offset_back_edge = frame.composite_t_ms - frame.back_edge_present_t_ms
        offsetslist_unique_otime_vs_ptime_front_edge.append(otime_vs_ptime_offset_front_edge)
        offsetslist_unique_otime_vs_ptime_back_edge.append(otime_vs_ptime_offset_back_edge)

        if prev_back_edge_time_gap is None:
            front_edge_deviation = None
            back_edge_deviation = None
        else:
            front_edge_deviation = front_edge_time_gap - prev_front_edge_time_gap
            back_edge_deviation = back_edge_time_gap - prev_back_edge_time_gap
            deviationslist_rel_unique_output_front_edge.append(front_edge_deviation)
            deviationslist_rel_unique_output_back_edge.append(back_edge_deviation)
            deviationslist_abs_unique_output_front_edge.append(abs(front_edge_deviation))
            deviationslist_abs_unique_output_back_edge.append(abs(back_edge_deviation))

            otime_vs_ptime_deviation_front_edge = otime_vs_ptime_offset_front_edge - prev_otime_vs_ptime_offset_front_edge
            otime_vs_ptime_deviation_back_edge = otime_vs_ptime_offset_back_edge - prev_otime_vs_ptime_offset_back_edge
            deviationslist_rel_unique_otime_vs_ptime_front_edge.append(otime_vs_ptime_deviation_front_edge)
            deviationslist_rel_unique_otime_vs_ptime_back_edge.append(otime_vs_ptime_deviation_back_edge)
            deviationslist_abs_unique_otime_vs_ptime_front_edge.append(abs(otime_vs_ptime_deviation_front_edge))
            deviationslist_abs_unique_otime_vs_ptime_back_edge.append(abs(otime_vs_ptime_deviation_back_edge))

        # Always update "previous_..." variables, for the next frame to use,
        # regardless of whether we calculated gap stats for *this* frame.
        prev_present_frame = frame.present_frame
        prev_capture_frame = frame.capture_frame
        prev_composite_frame = frame.composite_frame
        prev_composite_time = frame.composite_t_ms
        prev_front_edge_present_time = frame.present_t_ms
        prev_back_edge_present_time = frame.back_edge_present_t_ms
        prev_front_edge_time_gap = front_edge_time_gap
        prev_back_edge_time_gap = back_edge_time_gap
        prev_otime_vs_ptime_offset_front_edge = otime_vs_ptime_offset_front_edge
        prev_otime_vs_ptime_offset_back_edge = otime_vs_ptime_offset_back_edge

    composited_frames_count = len(obs.composited_framelist)
    unique_composited_frames_count = len(obs.unique_composited_framelist)
    unique_frame_percentage = (unique_composited_frames_count / composited_frames_count) * 100

    print("\n\n===== STATS =====")
    print(f"Presented frames: {len(presented_framelist)}")
    print(f"Captured frames: {len(captured_framelist)} ({len(captured_framelist) - unique_composited_frames_count} unused)")
    print(f"Composited/output frames: {len(obs.composited_framelist)} ({unique_composited_frames_count} unique) ({unique_frame_percentage:0.3f}% unique, {100 - unique_frame_percentage:0.3f}% doubled)")

    capture_duration_ms = (presented_framelist[-1].present_t_ms - presented_framelist[0].present_t_ms)
    capture_duration_seconds = capture_duration_ms / 1000
    capture_duration_minutes = capture_duration_seconds / 60
    capture_duration_hours = capture_duration_minutes / 60

    avg_fps = (len(presented_framelist) - 1) / capture_duration_seconds
    print(f"\nInput/game average FPS: {avg_fps:0.3f}")

    g_avg = statistics.mean(gaplist_present_front_edge_times)
    g_med = statistics.median(gaplist_present_front_edge_times)
    g_min = min(gaplist_present_front_edge_times)
    g_max = max(gaplist_present_front_edge_times)
    g_stddev = statistics.stdev(gaplist_present_front_edge_times)
    print(
        f"Input/game frame time gaps (front edge): {g_avg:0.3f} avg, {g_med:0.3f} med, {g_min:0.3f} min, {g_max:0.3f} max, {g_stddev:0.3f} stddev")

    g_avg = statistics.mean(gaplist_present_back_edge_times[1:])
    g_med = statistics.median(gaplist_present_back_edge_times[1:])
    g_min = min(gaplist_present_back_edge_times[1:])
    g_max = max(gaplist_present_back_edge_times[1:])
    g_stddev = statistics.stdev(gaplist_present_back_edge_times[1:])
    print(
        f"Input/game frame time gaps (back edge): {g_avg:0.3f} avg, {g_med:0.3f} med, {g_min:0.3f} min, {g_max:0.3f} max, {g_stddev:0.3f} stddev")


    g_avg = statistics.mean(deviationslist_rel_present_front_edge)
    g_med = statistics.median(deviationslist_rel_present_front_edge)
    g_min = min(deviationslist_rel_present_front_edge)
    g_max = max(deviationslist_rel_present_front_edge)
    g_stddev = statistics.stdev(deviationslist_rel_present_front_edge)
    g_sum = sum(deviationslist_rel_present_front_edge)
    print(
        f"\nInput/game frame-to-frame frametime deviations (relative) (front edge): {g_avg:0.3f} avg, {g_med:0.3f} med, {g_min:0.3f} min, {g_max:0.3f} max, {g_stddev:0.3f} stddev, {g_sum:0.3f} sum")

    g_avg = statistics.mean(deviationslist_rel_present_back_edge)
    g_med = statistics.median(deviationslist_rel_present_back_edge)
    g_min = min(deviationslist_rel_present_back_edge)
    g_max = max(deviationslist_rel_present_back_edge)
    g_stddev = statistics.stdev(deviationslist_rel_present_back_edge)
    g_sum = sum(deviationslist_rel_present_back_edge)
    print(
        f"Input/game frame-to-frame frametime deviations (relative) (back edge): {g_avg:0.3f} avg, {g_med:0.3f} med, {g_min:0.3f} min, {g_max:0.3f} max, {g_stddev:0.3f} stddev, {g_sum:0.3f} sum")

    g_avg = statistics.mean(deviationslist_abs_present_front_edge)
    g_med = statistics.median(deviationslist_abs_present_front_edge)
    g_min = min(deviationslist_abs_present_front_edge)
    g_max = max(deviationslist_abs_present_front_edge)
    g_stddev = statistics.stdev(deviationslist_abs_present_front_edge)
    g_sum = sum(deviationslist_abs_present_front_edge)
    g_dps = g_sum / capture_duration_seconds
    g_dpm = g_sum / capture_duration_minutes
    print(
        f"Input/game frame-to-frame frametime deviations (absolute) (front edge): {g_avg:0.3f} avg, {g_med:0.3f} med, {g_min:0.3f} min, {g_max:0.3f} max, {g_stddev:0.3f} stddev, {g_sum:0.3f} sum, {g_dps:0.3f} deviation/sec, {g_dpm:0.3f} deviation/min")

    g_avg = statistics.mean(deviationslist_abs_present_back_edge)
    g_med = statistics.median(deviationslist_abs_present_back_edge)
    g_min = min(deviationslist_abs_present_back_edge)
    g_max = max(deviationslist_abs_present_back_edge)
    g_stddev = statistics.stdev(deviationslist_abs_present_back_edge)
    g_sum = sum(deviationslist_abs_present_back_edge)
    g_dps = g_sum / capture_duration_seconds
    g_dpm = g_sum / capture_duration_minutes
    print(
        f"Input/game frame-to-frame frametime deviations (absolute) (back edge): {g_avg:0.3f} avg, {g_med:0.3f} med, {g_min:0.3f} min, {g_max:0.3f} max, {g_stddev:0.3f} stddev, {g_sum:0.3f} sum, {g_dps:0.3f} deviation/sec, {g_dpm:0.3f} deviation/min")


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


    g_avg = statistics.mean(deviationslist_rel_captured_front_edge)
    g_med = statistics.median(deviationslist_rel_captured_front_edge)
    g_min = min(deviationslist_rel_captured_front_edge)
    g_max = max(deviationslist_rel_captured_front_edge)
    g_stddev = statistics.stdev(deviationslist_rel_captured_front_edge)
    g_sum = sum(deviationslist_rel_captured_front_edge)
    print(
        f"\nCaptured frame-to-frame frametime deviations (relative) (front edge): {g_avg:0.3f} avg, {g_med:0.3f} med, {g_min:0.3f} min, {g_max:0.3f} max, {g_stddev:0.3f} stddev, {g_sum:0.3f} sum")

    g_avg = statistics.mean(deviationslist_rel_captured_back_edge)
    g_med = statistics.median(deviationslist_rel_captured_back_edge)
    g_min = min(deviationslist_rel_captured_back_edge)
    g_max = max(deviationslist_rel_captured_back_edge)
    g_stddev = statistics.stdev(deviationslist_rel_captured_back_edge)
    g_sum = sum(deviationslist_rel_captured_back_edge)
    print(
        f"Captured frame-to-frame frametime deviations (relative) (back edge): {g_avg:0.3f} avg, {g_med:0.3f} med, {g_min:0.3f} min, {g_max:0.3f} max, {g_stddev:0.3f} stddev, {g_sum:0.3f} sum")

    g_avg = statistics.mean(deviationslist_abs_captured_front_edge)
    g_med = statistics.median(deviationslist_abs_captured_front_edge)
    g_min = min(deviationslist_abs_captured_front_edge)
    g_max = max(deviationslist_abs_captured_front_edge)
    g_stddev = statistics.stdev(deviationslist_abs_captured_front_edge)
    g_sum = sum(deviationslist_abs_captured_front_edge)
    g_dps = g_sum / capture_duration_seconds
    g_dpm = g_sum / capture_duration_minutes
    print(
        f"Captured frame-to-frame frametime deviations (absolute) (front edge): {g_avg:0.3f} avg, {g_med:0.3f} med, {g_min:0.3f} min, {g_max:0.3f} max, {g_stddev:0.3f} stddev, {g_sum:0.3f} sum, {g_dps:0.3f} deviation/sec, {g_dpm:0.3f} deviation/min")

    g_avg = statistics.mean(deviationslist_abs_captured_back_edge)
    g_med = statistics.median(deviationslist_abs_captured_back_edge)
    g_min = min(deviationslist_abs_captured_back_edge)
    g_max = max(deviationslist_abs_captured_back_edge)
    g_stddev = statistics.stdev(deviationslist_abs_captured_back_edge)
    g_sum = sum(deviationslist_abs_captured_back_edge)
    g_dps = g_sum / capture_duration_seconds
    g_dpm = g_sum / capture_duration_minutes
    print(
        f"Captured frame-to-frame frametime deviations (absolute) (back edge): {g_avg:0.3f} avg, {g_med:0.3f} med, {g_min:0.3f} min, {g_max:0.3f} max, {g_stddev:0.3f} stddev, {g_sum:0.3f} sum, {g_dps:0.3f} deviation/sec, {g_dpm:0.3f} deviation/min")


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


    g_avg = statistics.mean(deviationslist_rel_output_front_edge)
    g_med = statistics.median(deviationslist_rel_output_front_edge)
    g_min = min(deviationslist_rel_output_front_edge)
    g_max = max(deviationslist_rel_output_front_edge)
    g_stddev = statistics.stdev(deviationslist_rel_output_front_edge)
    g_sum = sum(deviationslist_rel_output_front_edge)
    print(
        f"\nOutput/composited frame-to-frame frametime deviations (relative) (front edge): {g_avg:0.3f} avg, {g_med:0.3f} med, {g_min:0.3f} min, {g_max:0.3f} max, {g_stddev:0.3f} stddev, {g_sum:0.3f} sum")

    g_avg = statistics.mean(deviationslist_rel_output_back_edge)
    g_med = statistics.median(deviationslist_rel_output_back_edge)
    g_min = min(deviationslist_rel_output_back_edge)
    g_max = max(deviationslist_rel_output_back_edge)
    g_stddev = statistics.stdev(deviationslist_rel_output_back_edge)
    g_sum = sum(deviationslist_rel_output_back_edge)
    print(
        f"Output/composited frame-to-frame frametime deviations (relative) (back edge): {g_avg:0.3f} avg, {g_med:0.3f} med, {g_min:0.3f} min, {g_max:0.3f} max, {g_stddev:0.3f} stddev, {g_sum:0.3f} sum")

    g_avg = statistics.mean(deviationslist_abs_output_front_edge)
    g_med = statistics.median(deviationslist_abs_output_front_edge)
    g_min = min(deviationslist_abs_output_front_edge)
    g_max = max(deviationslist_abs_output_front_edge)
    g_stddev = statistics.stdev(deviationslist_abs_output_front_edge)
    g_sum = sum(deviationslist_abs_output_front_edge)
    g_dps = g_sum / capture_duration_seconds
    g_dpm = g_sum / capture_duration_minutes
    print(
        f"Output/composited frame-to-frame frametime deviations (absolute) (front edge): {g_avg:0.3f} avg, {g_med:0.3f} med, {g_min:0.3f} min, {g_max:0.3f} max, {g_stddev:0.3f} stddev, {g_sum:0.3f} sum, {g_dps:0.3f} deviation/sec, {g_dpm:0.3f} deviation/min")

    g_avg = statistics.mean(deviationslist_abs_output_back_edge)
    g_med = statistics.median(deviationslist_abs_output_back_edge)
    g_min = min(deviationslist_abs_output_back_edge)
    g_max = max(deviationslist_abs_output_back_edge)
    g_stddev = statistics.stdev(deviationslist_abs_output_back_edge)
    g_sum = sum(deviationslist_abs_output_back_edge)
    g_dps = g_sum / capture_duration_seconds
    g_dpm = g_sum / capture_duration_minutes
    print(
        f"Output/composited frame-to-frame frametime deviations (absolute) (back edge): {g_avg:0.3f} avg, {g_med:0.3f} med, {g_min:0.3f} min, {g_max:0.3f} max, {g_stddev:0.3f} stddev, {g_sum:0.3f} sum, {g_dps:0.3f} deviation/sec, {g_dpm:0.3f} deviation/min")

    g_avg = statistics.mean(offsetslist_otime_vs_ptime_front_edge)
    g_med = statistics.median(offsetslist_otime_vs_ptime_front_edge)
    g_min = min(offsetslist_otime_vs_ptime_front_edge)
    g_max = max(offsetslist_otime_vs_ptime_front_edge)
    g_stddev = statistics.stdev(offsetslist_otime_vs_ptime_front_edge)
    g_sum = sum(offsetslist_otime_vs_ptime_front_edge)
    g_dps = g_sum / capture_duration_seconds
    g_dpm = g_sum / capture_duration_minutes
    print(
        f"Output/composited time offsets (otime vs ptime) (front edge): {g_avg:0.3f} avg, {g_med:0.3f} med, {g_min:0.3f} min, {g_max:0.3f} max, {g_stddev:0.3f} stddev, {g_sum:0.3f} sum, {g_dps:0.3f} deviation/sec, {g_dpm:0.3f} deviation/min")

    g_avg = statistics.mean(offsetslist_otime_vs_ptime_back_edge)
    g_med = statistics.median(offsetslist_otime_vs_ptime_back_edge)
    g_min = min(offsetslist_otime_vs_ptime_back_edge)
    g_max = max(offsetslist_otime_vs_ptime_back_edge)
    g_stddev = statistics.stdev(offsetslist_otime_vs_ptime_back_edge)
    g_sum = sum(offsetslist_otime_vs_ptime_back_edge)
    g_dps = g_sum / capture_duration_seconds
    g_dpm = g_sum / capture_duration_minutes
    print(
        f"Output/composited time offsets (otime vs ptime) (back edge): {g_avg:0.3f} avg, {g_med:0.3f} med, {g_min:0.3f} min, {g_max:0.3f} max, {g_stddev:0.3f} stddev, {g_sum:0.3f} sum, {g_dps:0.3f} deviation/sec, {g_dpm:0.3f} deviation/min")

    g_avg = statistics.mean(deviationslist_rel_otime_vs_ptime_front_edge)
    g_med = statistics.median(deviationslist_rel_otime_vs_ptime_front_edge)
    g_min = min(deviationslist_rel_otime_vs_ptime_front_edge)
    g_max = max(deviationslist_rel_otime_vs_ptime_front_edge)
    g_stddev = statistics.stdev(deviationslist_rel_otime_vs_ptime_front_edge)
    g_sum = sum(deviationslist_rel_otime_vs_ptime_front_edge)
    g_dps = g_sum / capture_duration_seconds
    g_dpm = g_sum / capture_duration_minutes
    print(
        f"Output/composited time offset (otime vs ptime) frame-to-frame deviation (relative) (front edge): {g_avg:0.3f} avg, {g_med:0.3f} med, {g_min:0.3f} min, {g_max:0.3f} max, {g_stddev:0.3f} stddev, {g_sum:0.3f} sum")

    g_avg = statistics.mean(deviationslist_rel_otime_vs_ptime_back_edge)
    g_med = statistics.median(deviationslist_rel_otime_vs_ptime_back_edge)
    g_min = min(deviationslist_rel_otime_vs_ptime_back_edge)
    g_max = max(deviationslist_rel_otime_vs_ptime_back_edge)
    g_stddev = statistics.stdev(deviationslist_rel_otime_vs_ptime_back_edge)
    g_sum = sum(deviationslist_rel_otime_vs_ptime_back_edge)
    g_dps = g_sum / capture_duration_seconds
    g_dpm = g_sum / capture_duration_minutes
    print(
        f"Output/composited time offset (otime vs ptime) frame-to-frame deviation (relative) (back edge): {g_avg:0.3f} avg, {g_med:0.3f} med, {g_min:0.3f} min, {g_max:0.3f} max, {g_stddev:0.3f} stddev, {g_sum:0.3f} sum")

    g_avg = statistics.mean(deviationslist_abs_otime_vs_ptime_front_edge)
    g_med = statistics.median(deviationslist_abs_otime_vs_ptime_front_edge)
    g_min = min(deviationslist_abs_otime_vs_ptime_front_edge)
    g_max = max(deviationslist_abs_otime_vs_ptime_front_edge)
    g_stddev = statistics.stdev(deviationslist_abs_otime_vs_ptime_front_edge)
    g_sum = sum(deviationslist_abs_otime_vs_ptime_front_edge)
    g_dps = g_sum / capture_duration_seconds
    g_dpm = g_sum / capture_duration_minutes
    print(
        f"Output/composited time offset (otime vs ptime) frame-to-frame deviation (absolute) (front edge): {g_avg:0.3f} avg, {g_med:0.3f} med, {g_min:0.3f} min, {g_max:0.3f} max, {g_stddev:0.3f} stddev, {g_sum:0.3f} sum, {g_dps:0.3f} deviation/sec, {g_dpm:0.3f} deviation/min")

    g_avg = statistics.mean(deviationslist_abs_otime_vs_ptime_back_edge)
    g_med = statistics.median(deviationslist_abs_otime_vs_ptime_back_edge)
    g_min = min(deviationslist_abs_otime_vs_ptime_back_edge)
    g_max = max(deviationslist_abs_otime_vs_ptime_back_edge)
    g_stddev = statistics.stdev(deviationslist_abs_otime_vs_ptime_back_edge)
    g_sum = sum(deviationslist_abs_otime_vs_ptime_back_edge)
    g_dps = g_sum / capture_duration_seconds
    g_dpm = g_sum / capture_duration_minutes
    print(
        f"Output/composited time offset (otime vs ptime) frame-to-frame deviation (absolute) (back edge): {g_avg:0.3f} avg, {g_med:0.3f} med, {g_min:0.3f} min, {g_max:0.3f} max, {g_stddev:0.3f} stddev, {g_sum:0.3f} sum, {g_dps:0.3f} deviation/sec, {g_dpm:0.3f} deviation/min")


    g_avg = statistics.mean(gaplist_unique_output_frames)
    g_med = statistics.median(gaplist_unique_output_frames)
    g_min = min(gaplist_unique_output_frames)
    g_max = max(gaplist_unique_output_frames)
    g_stddev = statistics.stdev(gaplist_unique_output_frames)
    print(
        f"\nUnique Output/composited frame pframe number gaps: {g_avg:0.2f} avg, {g_med:0.2f} med, {g_min} min, {g_max} max, {g_stddev:0.2f} stddev")

    g_avg = statistics.mean(gaplist_unique_output_frame_oframes)
    g_med = statistics.median(gaplist_unique_output_frame_oframes)
    g_min = min(gaplist_unique_output_frame_oframes)
    g_max = max(gaplist_unique_output_frame_oframes)
    g_stddev = statistics.stdev(gaplist_unique_output_frame_oframes)
    print(
        f"Unique Output/composited frame oframe number gaps: {g_avg:0.2f} avg, {g_med:0.2f} med, {g_min} min, {g_max} max, {g_stddev:0.2f} stddev")

    g_avg = statistics.mean(gaplist_unique_output_front_edge_times)
    g_med = statistics.median(gaplist_unique_output_front_edge_times)
    g_min = min(gaplist_unique_output_front_edge_times)
    g_max = max(gaplist_unique_output_front_edge_times)
    g_stddev = statistics.stdev(gaplist_unique_output_front_edge_times)
    print(
        f"Unique Output/composited frame time gaps (front edge): {g_avg:0.3f} avg, {g_med:0.3f} med, {g_min:0.3f} min, {g_max:0.3f} max, {g_stddev:0.3f} stddev")

    g_avg = statistics.mean(gaplist_unique_output_back_edge_times)
    g_med = statistics.median(gaplist_unique_output_back_edge_times)
    g_min = min(gaplist_unique_output_back_edge_times)
    g_max = max(gaplist_unique_output_back_edge_times)
    g_stddev = statistics.stdev(gaplist_unique_output_back_edge_times)
    print(
        f"Unique Output/composited frame time gaps (back edge): {g_avg:0.3f} avg, {g_med:0.3f} med, {g_min:0.3f} min, {g_max:0.3f} max, {g_stddev:0.3f} stddev")

    g_avg = statistics.mean(gaplist_unique_output_composite_times)
    g_med = statistics.median(gaplist_unique_output_composite_times)
    g_min = min(gaplist_unique_output_composite_times)
    g_max = max(gaplist_unique_output_composite_times)
    g_stddev = statistics.stdev(gaplist_unique_output_composite_times)
    print(
        f"Unique Output/composited frame time gaps (otime): {g_avg:0.3f} avg, {g_med:0.3f} med, {g_min:0.3f} min, {g_max:0.3f} max, {g_stddev:0.3f} stddev")


    g_avg = statistics.mean(deviationslist_rel_unique_output_front_edge)
    g_med = statistics.median(deviationslist_rel_unique_output_front_edge)
    g_min = min(deviationslist_rel_unique_output_front_edge)
    g_max = max(deviationslist_rel_unique_output_front_edge)
    g_stddev = statistics.stdev(deviationslist_rel_unique_output_front_edge)
    g_sum = sum(deviationslist_rel_unique_output_front_edge)
    print(
        f"\nUnique Output/composited frame-to-frame frametime deviations (relative) (front edge): {g_avg:0.3f} avg, {g_med:0.3f} med, {g_min:0.3f} min, {g_max:0.3f} max, {g_stddev:0.3f} stddev, {g_sum:0.3f} sum")

    g_avg = statistics.mean(deviationslist_rel_unique_output_back_edge)
    g_med = statistics.median(deviationslist_rel_unique_output_back_edge)
    g_min = min(deviationslist_rel_unique_output_back_edge)
    g_max = max(deviationslist_rel_unique_output_back_edge)
    g_stddev = statistics.stdev(deviationslist_rel_unique_output_back_edge)
    g_sum = sum(deviationslist_rel_unique_output_back_edge)
    print(
        f"Unique Output/composited frame-to-frame frametime deviations (relative) (back edge): {g_avg:0.3f} avg, {g_med:0.3f} med, {g_min:0.3f} min, {g_max:0.3f} max, {g_stddev:0.3f} stddev, {g_sum:0.3f} sum")

    g_avg = statistics.mean(deviationslist_abs_unique_output_front_edge)
    g_med = statistics.median(deviationslist_abs_unique_output_front_edge)
    g_min = min(deviationslist_abs_unique_output_front_edge)
    g_max = max(deviationslist_abs_unique_output_front_edge)
    g_stddev = statistics.stdev(deviationslist_abs_unique_output_front_edge)
    g_sum = sum(deviationslist_abs_unique_output_front_edge)
    g_dps = g_sum / capture_duration_seconds
    g_dpm = g_sum / capture_duration_minutes
    print(
        f"Unique Output/composited frame-to-frame frametime deviations (absolute) (front edge): {g_avg:0.3f} avg, {g_med:0.3f} med, {g_min:0.3f} min, {g_max:0.3f} max, {g_stddev:0.3f} stddev, {g_sum:0.3f} sum, {g_dps:0.3f} deviation/sec, {g_dpm:0.3f} deviation/min")

    g_avg = statistics.mean(deviationslist_abs_unique_output_back_edge)
    g_med = statistics.median(deviationslist_abs_unique_output_back_edge)
    g_min = min(deviationslist_abs_unique_output_back_edge)
    g_max = max(deviationslist_abs_unique_output_back_edge)
    g_stddev = statistics.stdev(deviationslist_abs_unique_output_back_edge)
    g_sum = sum(deviationslist_abs_unique_output_back_edge)
    g_dps = g_sum / capture_duration_seconds
    g_dpm = g_sum / capture_duration_minutes
    print(
        f"Unique Output/composited frame-to-frame frametime deviations (absolute) (back edge): {g_avg:0.3f} avg, {g_med:0.3f} med, {g_min:0.3f} min, {g_max:0.3f} max, {g_stddev:0.3f} stddev, {g_sum:0.3f} sum, {g_dps:0.3f} deviation/sec, {g_dpm:0.3f} deviation/min")

    g_avg = statistics.mean(offsetslist_unique_otime_vs_ptime_front_edge)
    g_med = statistics.median(offsetslist_unique_otime_vs_ptime_front_edge)
    g_min = min(offsetslist_unique_otime_vs_ptime_front_edge)
    g_max = max(offsetslist_unique_otime_vs_ptime_front_edge)
    g_stddev = statistics.stdev(offsetslist_unique_otime_vs_ptime_front_edge)
    g_sum = sum(offsetslist_unique_otime_vs_ptime_front_edge)
    g_dps = g_sum / capture_duration_seconds
    g_dpm = g_sum / capture_duration_minutes
    print(
        f"Unique Output/composited time offsets (otime vs ptime) (front edge): {g_avg:0.3f} avg, {g_med:0.3f} med, {g_min:0.3f} min, {g_max:0.3f} max, {g_stddev:0.3f} stddev, {g_sum:0.3f} sum, {g_dps:0.3f} deviation/sec, {g_dpm:0.3f} deviation/min")

    g_avg = statistics.mean(offsetslist_unique_otime_vs_ptime_back_edge)
    g_med = statistics.median(offsetslist_unique_otime_vs_ptime_back_edge)
    g_min = min(offsetslist_unique_otime_vs_ptime_back_edge)
    g_max = max(offsetslist_unique_otime_vs_ptime_back_edge)
    g_stddev = statistics.stdev(offsetslist_unique_otime_vs_ptime_back_edge)
    g_sum = sum(offsetslist_unique_otime_vs_ptime_back_edge)
    g_dps = g_sum / capture_duration_seconds
    g_dpm = g_sum / capture_duration_minutes
    print(
        f"Unique Output/composited time offsets (otime vs ptime) (back edge): {g_avg:0.3f} avg, {g_med:0.3f} med, {g_min:0.3f} min, {g_max:0.3f} max, {g_stddev:0.3f} stddev, {g_sum:0.3f} sum, {g_dps:0.3f} deviation/sec, {g_dpm:0.3f} deviation/min")

    g_avg = statistics.mean(deviationslist_rel_unique_otime_vs_ptime_front_edge)
    g_med = statistics.median(deviationslist_rel_unique_otime_vs_ptime_front_edge)
    g_min = min(deviationslist_rel_unique_otime_vs_ptime_front_edge)
    g_max = max(deviationslist_rel_unique_otime_vs_ptime_front_edge)
    g_stddev = statistics.stdev(deviationslist_rel_unique_otime_vs_ptime_front_edge)
    g_sum = sum(deviationslist_rel_unique_otime_vs_ptime_front_edge)
    g_dps = g_sum / capture_duration_seconds
    g_dpm = g_sum / capture_duration_minutes
    print(
        f"Unique Output/composited time offset (otime vs ptime) frame-to-frame deviation (relative) (front edge): {g_avg:0.3f} avg, {g_med:0.3f} med, {g_min:0.3f} min, {g_max:0.3f} max, {g_stddev:0.3f} stddev, {g_sum:0.3f} sum")

    g_avg = statistics.mean(deviationslist_rel_unique_otime_vs_ptime_back_edge)
    g_med = statistics.median(deviationslist_rel_unique_otime_vs_ptime_back_edge)
    g_min = min(deviationslist_rel_unique_otime_vs_ptime_back_edge)
    g_max = max(deviationslist_rel_unique_otime_vs_ptime_back_edge)
    g_stddev = statistics.stdev(deviationslist_rel_unique_otime_vs_ptime_back_edge)
    g_sum = sum(deviationslist_rel_unique_otime_vs_ptime_back_edge)
    g_dps = g_sum / capture_duration_seconds
    g_dpm = g_sum / capture_duration_minutes
    print(
        f"Unique Output/composited time offset (otime vs ptime) frame-to-frame deviation (relative) (back edge): {g_avg:0.3f} avg, {g_med:0.3f} med, {g_min:0.3f} min, {g_max:0.3f} max, {g_stddev:0.3f} stddev, {g_sum:0.3f} sum")

    g_avg = statistics.mean(deviationslist_abs_unique_otime_vs_ptime_front_edge)
    g_med = statistics.median(deviationslist_abs_unique_otime_vs_ptime_front_edge)
    g_min = min(deviationslist_abs_unique_otime_vs_ptime_front_edge)
    g_max = max(deviationslist_abs_unique_otime_vs_ptime_front_edge)
    g_stddev = statistics.stdev(deviationslist_abs_unique_otime_vs_ptime_front_edge)
    g_sum = sum(deviationslist_abs_unique_otime_vs_ptime_front_edge)
    g_dps = g_sum / capture_duration_seconds
    g_dpm = g_sum / capture_duration_minutes
    print(
        f"Unique Output/composited time offset (otime vs ptime) frame-to-frame deviation (absolute) (front edge): {g_avg:0.3f} avg, {g_med:0.3f} med, {g_min:0.3f} min, {g_max:0.3f} max, {g_stddev:0.3f} stddev, {g_sum:0.3f} sum, {g_dps:0.3f} deviation/sec, {g_dpm:0.3f} deviation/min")

    g_avg = statistics.mean(deviationslist_abs_unique_otime_vs_ptime_back_edge)
    g_med = statistics.median(deviationslist_abs_unique_otime_vs_ptime_back_edge)
    g_min = min(deviationslist_abs_unique_otime_vs_ptime_back_edge)
    g_max = max(deviationslist_abs_unique_otime_vs_ptime_back_edge)
    g_stddev = statistics.stdev(deviationslist_abs_unique_otime_vs_ptime_back_edge)
    g_sum = sum(deviationslist_abs_unique_otime_vs_ptime_back_edge)
    g_dps = g_sum / capture_duration_seconds
    g_dpm = g_sum / capture_duration_minutes
    print(
        f"Unique Output/composited time offset (otime vs ptime) frame-to-frame deviation (absolute) (back edge): {g_avg:0.3f} avg, {g_med:0.3f} med, {g_min:0.3f} min, {g_max:0.3f} max, {g_stddev:0.3f} stddev, {g_sum:0.3f} sum, {g_dps:0.3f} deviation/sec, {g_dpm:0.3f} deviation/min")

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
