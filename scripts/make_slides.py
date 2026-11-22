"""Build slides/outrun.pptx (python-pptx) and slides/outrun.pdf (same layout via HTML + headless Chrome).

Content comes from REPORT.md, reports/metrics and reports/figures; screenshots are slides/img/app_*.png.
LibreOffice is not installed, so the PDF is rendered from the same slide spec (absolute inch/pt coordinates).
"""

import base64
import html as H
import subprocess
import tempfile
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
root = Path(__file__).resolve().parent.parent
out = root / "slides"
W, HT = 13.333, 7.5
NAVY, ORANGE, INK, GREY, RED, GREEN, LIGHT = (
    "16202c",
    "e8590c",
    "1f2933",
    "5f6b7a",
    "b42318",
    "1e7a3c",
    "eef1f5",
)
FONT = "Arial"


def img(name):
    return str(root / ("reports/figures/" + name if not name.startswith("app_") else "slides/img/" + name))


# ---- slide spec -----------------------------------------------------------------------------
# element kinds: text(x,y,w,h,paras,size,color,bold,align), bullets(...items), table(...), image(...), rect(...)
S = []


def slide(title, notes, *els, dark=False):
    S.append(dict(title=title, notes=notes, els=list(els), dark=dark))


def T(x, y, w, h, paras, size=16, color=INK, bold=False, align="l"):
    return dict(
        k="text",
        x=x,
        y=y,
        w=w,
        h=h,
        paras=paras if isinstance(paras, list) else [paras],
        size=size,
        color=color,
        bold=bold,
        align=align,
    )


def B(x, y, w, h, items, size=16, color=INK):
    return dict(k="bul", x=x, y=y, w=w, h=h, items=items, size=size, color=color)


def TB(x, y, cols, rows, size=11, rh=0.4, hl=()):
    return dict(k="tbl", x=x, y=y, cols=cols, rows=rows, size=size, rh=rh, hl=hl)


def IM(x, y, w, h, name):
    return dict(k="img", x=x, y=y, w=w, h=h, name=name)


def RC(x, y, w, h, fill):
    return dict(k="rect", x=x, y=y, w=w, h=h, fill=fill)


slide(
    "Outrun",
    "Welcome. Outrun is my AI final project for CMSC-5713. In one sentence, it is the approved proposal: an AI system that "
    "uses deep learning and real weather, radar and satellite data to forecast six severe-weather hazards, then uses A* "
    "search and optimization on real road maps to plan safe evacuations, shown in an interactive map app. Everything in "
    "this talk comes from a single held-out evaluation, run once, and I will tell you where the models lost as well as where they won.",
    T(0.9, 2.2, 11.5, 1.2, "Outrun", 60, "ffffff", True),
    T(
        0.9,
        3.4,
        11.5,
        1.2,
        "Forecasting six severe-weather hazards and planning safe evacuation routes on real road networks",
        26,
        "dfe6ee",
    ),
    T(0.9, 5.6, 11.5, 0.5, "CMSC-5713 AI final project  |  Milton Adina  |  December 2026", 16, "9fb0c3"),
    RC(0.9, 4.9, 1.6, 0.07, ORANGE),
    dark=True,
)

slide(
    "Oklahoma faces six hazards, and each needs a different forecast",
    "Oklahoma gets hit by severe storms, tornadoes, flash floods, wildfire, ice storms and smoke. Each one has its own "
    "data and its own failure mode, but a person trying to leave has one question: which road is safe, right now, in the next hour? "
    "The proposal says: use deep learning and real weather, radar and satellite data to forecast the hazards, then use A* search "
    "and optimization on real road maps to plan safe evacuations, shelters, whole-city evacuation and emergency-vehicle routes, "
    "in an interactive map app. That sentence is the whole project.",
    B(
        0.7,
        1.6,
        6.0,
        4.6,
        [
            "Severe storms (radar nowcast)",
            "Tornadoes (radar chips)",
            "Flash floods (rain and runoff)",
            "Wildfire spread (satellite fire masks)",
            "Icy roads (winter weather)",
            "Smoke (PM2.5)",
        ],
        22,
    ),
    RC(7.1, 1.6, 5.5, 4.4, LIGHT),
    T(7.4, 1.8, 5.0, 0.5, "The approved proposal", 14, ORANGE, True),
    T(
        7.4,
        2.3,
        4.9,
        3.6,
        "An AI system that uses deep learning and real weather, radar and satellite data to forecast severe storms, "
        "tornadoes, flash floods, wildfire spread, icy roads and smoke, then uses A* search and optimization on real road maps "
        "to plan safe evacuation routes, shelter assignments, whole-city evacuations and emergency-vehicle routes, "
        "shown in an interactive map app.",
        17,
        INK,
    ),
)

slide(
    "What I built: forecast, plan, show",
    "Three parts. Forecasting: six hazard models, each producing a calibrated risk layer on one shared one-kilometre grid. "
    "Planning: my own time-dependent A* on real OpenStreetMap roads, a min-cost-flow shelter assignment, simulated annealing "
    "for whole-metro evacuation compared with MSA user equilibrium plus shortest and safest-path baselines, and Hungarian dispatch "
    "for emergency vehicles. App: FastAPI plus a vanilla MapLibre front end that runs fully offline from committed scenario bundles.",
    RC(0.7, 1.7, 3.9, 4.4, LIGHT),
    RC(4.7, 1.7, 3.9, 4.4, LIGHT),
    RC(8.7, 1.7, 3.9, 4.4, LIGHT),
    T(0.9, 1.85, 3.5, 0.5, "1. Forecast", 22, ORANGE, True),
    T(
        0.9,
        2.5,
        3.5,
        3.5,
        "Six hazard models, each a calibrated risk layer on one shared 1 km grid (EPSG:5070).",
        18,
    ),
    T(4.9, 1.85, 3.5, 0.5, "2. Plan", 22, ORANGE, True),
    B(
        4.9,
        2.5,
        3.5,
        3.5,
        [
            "Time-dependent A* on real OSM roads",
            "Min-cost-flow shelter assignment",
            "Simulated annealing for whole-city evacuation",
            "Hungarian dispatch for emergency vehicles",
        ],
        17,
    ),
    T(8.9, 1.85, 3.5, 0.5, "3. Show", 22, ORANGE, True),
    T(
        8.9,
        2.5,
        3.5,
        3.5,
        "FastAPI backend and a vanilla MapLibre front end. Runs fully offline from committed scenario bundles.",
        18,
    ),
    T(
        0.7,
        6.4,
        11.9,
        0.5,
        "Data: SEVIR radar and GOES-16, TorNet, HRRR, MRMS/FLASH, NDWS, WildfireSpreadTS, NOAA HMS, AQS, OpenStreetMap",
        13,
        GREY,
    ),
)

slide(
    "Architecture: six models feed one RiskStack, which feeds one router",
    "Six models feed one RiskStack, which feeds one router. The key design choice is the shared interface. Every hazard "
    "is a RiskLayer on the same grid, and the planners only see the stack, so they do not care which hazard they are routing around. "
    "What you see on the map is what the router uses.",
    T(0.7, 1.7, 2.6, 0.45, "Data", 14, GREY, True),
    RC(0.7, 2.2, 2.6, 3.6, LIGHT),
    B(
        0.85,
        2.3,
        2.4,
        3.4,
        [
            "Radar: SEVIR VIL, TorNet",
            "Satellite: GOES-16 IR107, MODIS, VIIRS, HMS",
            "HRRR, MRMS/FLASH",
            "Storm Events, NWS zones, AQS PM2.5",
            "OpenStreetMap roads",
        ],
        13,
    ),
    T(3.6, 3.7, 0.5, 0.5, ">", 28, ORANGE, True),
    T(4.1, 1.7, 2.6, 0.45, "Six hazard models", 14, GREY, True),
    RC(4.1, 2.2, 2.6, 3.6, LIGHT),
    B(
        4.25,
        2.3,
        2.4,
        3.4,
        ["Storm U-Net", "Tornado CNN", "Flood U-Net", "Wildfire CNN", "Ice MIL-CNN", "Smoke model"],
        14,
    ),
    T(7.0, 3.7, 0.5, 0.5, ">", 28, ORANGE, True),
    T(7.5, 1.7, 2.6, 0.45, "RiskStack", 14, GREY, True),
    RC(7.5, 2.2, 2.6, 3.6, "fde8d9"),
    T(
        7.65,
        2.4,
        2.3,
        3.2,
        ["RiskLayer per hazard", "one 1 km grid", "calibrated probabilities", "HazardPolicy per hazard"],
        14,
    ),
    T(10.4, 3.7, 0.5, 0.5, ">", 28, ORANGE, True),
    T(10.9, 1.7, 2.0, 0.45, "Planners", 14, GREY, True),
    RC(10.9, 2.2, 2.0, 3.6, LIGHT),
    B(11.0, 2.3, 1.85, 3.4, ["A* router", "Shelter flow", "Evacuation SA", "EMS dispatch"], 13),
    T(
        0.7,
        6.2,
        12.0,
        0.5,
        "FastAPI (/api, SSE)  <->  vanilla MapLibre GL JS with local PMTiles; scenario bundles make it run offline",
        14,
        GREY,
    ),
)

slide(
    "One interface for every hazard: RiskLayer and HazardPolicy",
    "A RiskLayer is a calibrated probability map on the shared grid with a time axis. A HazardPolicy says how that risk "
    "turns into routing cost: whether it blocks an edge, how much it slows travel, and how heavily it is weighted. Cost on an "
    "edge is travel time times one plus the weighted sum of the hazard risks. A lambda slider in the app scales those weights from "
    "safer to faster. Because the router only sees this cost, adding a seventh hazard would not change the planners.",
    RC(0.7, 1.7, 5.9, 4.5, LIGHT),
    T(0.9, 1.85, 5.5, 0.45, "RiskLayer", 20, ORANGE, True),
    B(
        0.9,
        2.4,
        5.5,
        3.7,
        [
            "Calibrated probability map on the 1 km grid, with a time axis",
            "Produced by a model or by a baseline, so both are scored the same way",
            "Collected in a RiskStack that the planners read",
        ],
        18,
    ),
    RC(6.9, 1.7, 5.7, 4.5, LIGHT),
    T(7.1, 1.85, 5.3, 0.45, "HazardPolicy", 20, ORANGE, True),
    B(
        7.1,
        2.4,
        5.3,
        3.7,
        [
            "Block threshold: a probability above it closes an edge",
            "Speed factor: slows the clock as risk rises",
            "Exposure weight in the edge cost: tau x (1 + sum of weight x risk)",
            "A lambda slider scales the weights, safer to faster",
        ],
        18,
    ),
    T(0.7, 6.5, 12.0, 0.5, "Planners never see which hazard they route around", 16, INK, True),
)

slide(
    "Six models against baselines: four wins, two non-wins, one loss",
    "This is the held-out test table, evaluated once. Intervals are 95 percent bootstrap CIs. The deep model wins on storm, "
    "flood, wildfire on NDWS and smoke RMSE. Tornado is a tie at 0.761 against 0.762, and below the published TorNet number of 0.874. "
    "Ice does not win. And on wildfire WSTS 2021 the deep model loses clearly, 0.372 against 0.456.",
    TB(
        0.6,
        1.55,
        [2.3, 2.3, 2.4, 3.35, 2.0],
        [
            ["Hazard", "Metric", "Deep model", "Best baseline", "Verdict"],
            ["Storm", "CSI mean 74/133", "0.533 [0.520, 0.544]", "0.467 [0.455, 0.477] pySTEPS", "DL wins"],
            ["Tornado", "AUC vs warned null", "0.761 [0.709, 0.810]", "0.762 [0.709, 0.811] raw CNN", "Tie"],
            ["Flood", "AUPRC", "0.224 [0.191, 0.255]", "0.144 [0.119, 0.167] logistic", "DL wins"],
            ["Wildfire NDWS", "PR-AUC", "0.357 [0.340, 0.373]", "0.153 [0.139, 0.168] logistic", "DL wins"],
            [
                "Wildfire WSTS 2021",
                "PR-AUC",
                "0.372 [0.323, 0.410]",
                "0.456 [0.416, 0.488] logistic",
                "DL loses",
            ],
            ["Ice", "Zone AUPRC", "0.801 [0.622, 0.858]", "0.804 [0.587, 0.876] logistic", "No win"],
            [
                "Smoke",
                "RMSE (lower better)",
                "10.209 [8.775, 11.951]",
                "12.198 [10.679, 13.913] HRRR",
                "DL wins",
            ],
        ],
        14,
        0.52,
        hl=((5, 4, RED), (1, 4, GREEN), (3, 4, GREEN), (4, 4, GREEN), (7, 4, GREEN)),
    ),
    T(
        0.6,
        5.9,
        12.1,
        0.9,
        "Test data was evaluated once, in T24. Model selection and calibration used validation data only. "
        "Intervals are 95% bootstrap CIs. NDWS intervals resample per sample and are optimistic.",
        14,
        GREY,
    ),
)

slide(
    "Where the deep model did not win",
    "I want to be plain about this. Tornado: calibration helped CSI and Brier but not ranking, and my AUC of 0.761 is below "
    "the published 0.874. Ice: the difference to logistic is minus 0.003 and indistinguishable. Wildfire: the NDWS win does "
    "not transfer to the out-of-time WSTS 2021 split, where even persistence at 0.433 beats my 0.372. Smoke: better RMSE, "
    "but persistence has better CSI, 0.182 against 0.114, and much higher POD.",
    IM(0.5, 1.5, 6.3, 3.3, "models_vs_baselines.png"),
    T(0.6, 4.95, 6.1, 1.0, "Figure: deep model against best baseline on the held-out test set.", 12, GREY),
    B(
        7.1,
        1.5,
        5.8,
        5.4,
        [
            "Tornado: AUC 0.761 vs 0.762 for its uncalibrated twin; published TorNet 0.874 not reached. Calibration helps CSI and Brier, not ranking",
            "Ice: 0.801 vs logistic 0.804; difference -0.003 [-0.045, 0.077], indistinguishable",
            "Wildfire WSTS 2021: 0.372 loses to logistic 0.456 and to persistence 0.433. The NDWS win does not transfer",
            "Smoke: better RMSE, but persistence wins CSI 0.182 vs 0.114 and POD 0.308 vs 0.122",
        ],
        16,
    ),
)

slide(
    "Satellite data: named, used, and measurably no help here",
    "The proposal calls for real satellite data, so it is in three places. GOES-16 IR 10.7 micron from SEVIR is stacked with "
    "radar VIL in the storm U-Net. MODIS fire masks and VIIRS bands feed the wildfire model. NOAA HMS smoke polygons feed the smoke model. "
    "I ran ablations for each. The honest result is that IR, FLASH and HMS added nothing measurable at this data size. "
    "The VIL-only storm model scores 0.536 against 0.533 with IR.",
    B(
        0.7,
        1.6,
        4.9,
        3.6,
        [
            "Storm: GOES-16 IR 10.7 um (IR107) from SEVIR, stacked with radar VIL",
            "Wildfire: MODIS fire masks and VIIRS NDVI/bands inside NDWS and WSTS",
            "Smoke: NOAA HMS satellite smoke polygons as a model input",
        ],
        16,
    ),
    TB(
        5.9,
        1.7,
        [1.5, 2.2, 3.2],
        [
            ["Ablation", "With", "Without"],
            ["Storm CSI", "0.533 (VIL+IR)", "0.536 [0.522, 0.548] (VIL only)"],
            ["Flood AUPRC", "0.224", "0.207 [0.175, 0.236] (no FLASH)"],
            ["Smoke RMSE", "10.209", "10.283 (no HMS)"],
        ],
        12,
        0.7,
    ),
    RC(0.7, 5.3, 11.9, 1.2, LIGHT),
    T(
        0.9,
        5.4,
        11.5,
        1.0,
        "Result: the satellite channels did not measurably help any model here. Differences are within the intervals. "
        "Finer or more channels are untested, so I do not conclude that satellite data cannot help.",
        16,
        INK,
        True,
    ),
)

slide(
    "Routing and optimization: five planners and five evacuation methods",
    "Routing is evaluated with four planners, all scored on the observed risk. Plain shortest time, a risk-aware planner with risk "
    "frozen at the start, a risk-aware planner on the baseline forecast, one on the deep forecast, and an oracle that plans on observed risk. "
    "For evacuation of the May 20 metro, simulated annealing has the lowest vehicle-hours, 1892 against 9417, and lower exposure than shortest "
    "and safest. MSA has by far the lowest exposure but takes 95 minutes to clear. The gap to the lower bound is still large, 264 percent for SA.",
    B(
        0.6,
        1.5,
        5.1,
        4.4,
        [
            "(a) shortest time",
            "(a') risk-aware, risk frozen at t0",
            "(b) risk-aware on baseline forecast",
            "(c) risk-aware on deep forecast",
            "(d) oracle on observed risk",
            "Scored on observed risk. Time-dependent A*, min-cost flow, simulated annealing with BPR congestion",
        ],
        16,
    ),
    TB(
        5.9,
        1.6,
        [1.6, 1.7, 2.0, 1.9],
        [
            ["Method", "Vehicle-hours", "Exposure (veh-h)", "Clearance (min)"],
            ["shortest", "9416.5", "3344.8", "32.5"],
            ["safest", "9416.9", "3344.8", "32.5"],
            ["MSA-UE", "2962.0", "814.1", "94.8"],
            ["SA", "1891.8", "2779.2", "33.3"],
            ["SA (lambda 0)", "2273.0", "3155.1", "27.1"],
        ],
        14,
        0.45,
    ),
    T(
        5.9,
        4.5,
        7.2,
        2.3,
        [
            "may20 metro, 5 seeds.",
            "SA: lowest vehicle-hours and lower exposure than shortest/safest, similar clearance. MSA: lowest exposure, slow clearance.",
            "Still unsolved: gap to the lower bound (2414) is 264% for SA. Shelters: 112,461 persons unserved (capacity-limited).",
        ],
        14,
        INK,
    ),
)

slide(
    "Gap closed: 40% in one storm, about zero on ice",
    "Gap closed measures how much of the distance between plain shortest path and the oracle the deep forecast recovers. On "
    "May 20 2019 it is 40 percent, with an interval from 17.6 to 56.1. But this is one scenario and it is not uniform: El Reno 16.7, "
    "flood 90 but with an interval past 100 so suggestive only, ice slightly negative, and smoke undefined because there is no routable risk. "
    "And against the frozen-at-t0 planner, the deep forecast is barely different, so most of the benefit comes from using any risk at all.",
    TB(
        0.5,
        1.6,
        [2.2, 0.8, 0.8, 0.8, 2.6],
        [
            ["Scenario", "(a)", "(c)", "(d)", "Gap closed (c)"],
            ["may20-2019", "5.903", "5.681", "5.349", "40.0% [17.6%, 56.1%]"],
            ["elreno-2019", "3.146", "2.920", "1.793", "16.7% [4.9%, 31.0%]"],
            ["flood-okc-2025", "1.156", "0.798", "0.758", "90.0% [32.8%, 121.8%]"],
            ["ice-okc-2025", "7.850", "7.856", "7.362", "-1.3% [-5.2%, -0.3%]"],
            ["smoke-okc-2024", "0.000", "0.000", "0.000", "n/a (NaN)"],
        ],
        12,
        0.5,
        hl=((1, 4, GREEN),),
    ),
    T(
        0.5,
        4.8,
        7.2,
        2.0,
        [
            "Mean observed exposure in risk-minutes.",
            "Flood: 95 paired origins, CI runs past 100%, suggestive only. Smoke: no routable risk.",
            "Against the frozen-at-t0 planner (a'), (c) barely differs: most of the benefit is risk-aware planning itself.",
        ],
        12,
        GREY,
    ),
    IM(8.0, 1.5, 5.0, 3.0, "gap_closed.png"),
    IM(8.0, 4.6, 5.0, 2.4, "routing_exposure.png"),
)

slide(
    "Demo: the app, running fully offline",
    "This is the app, served from committed scenario bundles with no network. The forecast tab shows the model layer on the map "
    "with a time slider and a Forecast versus Observed swipe. The evacuation tab runs simulated annealing live and animates vehicles by wave. "
    "Note the numbers in the app are a single run, so they differ slightly from the five-seed means in the report. The basemap tiles "
    "are not shown here; the overlays are the model output. In the live demo I walk El Reno 2019, shelters, the evacuation, and a flood dispatch.",
    IM(0.5, 1.5, 6.1, 3.43, "app_forecast.png"),
    IM(6.75, 1.5, 6.1, 3.43, "app_evac.png"),
    T(
        0.5,
        5.0,
        6.1,
        0.9,
        "Forecast tab: May 20 2019 storm layer, time slider, Forecast | Observed swipe, model card",
        13,
        GREY,
    ),
    T(
        6.75,
        5.0,
        6.1,
        0.9,
        "Evacuation tab: live simulated annealing, wave animation, KPIs. One run; the report table is a 5-seed mean",
        13,
        GREY,
    ),
    T(
        0.5,
        6.1,
        12.3,
        0.8,
        "make demo runs with no data, models or network. Six tabs: Forecast, Safe route, Shelters, Evacuation, Dispatch, Metrics.",
        16,
        INK,
        True,
    ),
)

slide(
    "Honest results: what held up and what did not",
    "To summarise the results. Wins: storm, flood, wildfire NDWS and smoke RMSE. Non-wins: tornado, ice, wildfire WSTS and smoke CSI, "
    "plus satellite channels that added nothing. Routing closed 40 percent of the gap in one storm and none on ice. "
    "The tornado demo never blocks a road, the May 20 tornado layer over Oklahoma City is empty and I did not track down why, and the El Reno "
    "timing slack is demo-only. Acceptance: 23 pass, 2 expected failures where I recorded the real number, and one skipped.",
    RC(0.6, 1.6, 5.9, 4.6, "e6f4ea"),
    T(0.8, 1.7, 5.5, 0.5, "Held up", 20, GREEN, True),
    B(
        0.8,
        2.3,
        5.5,
        3.8,
        [
            "Storm CSI 0.533 vs pySTEPS 0.467",
            "Flood AUPRC 0.224 vs 0.144",
            "Wildfire NDWS PR-AUC 0.357 vs 0.153",
            "Smoke RMSE 10.209 vs 12.198",
            "Routing gap closed 40.0% on may20-2019",
            "SA vehicle-hours 1892 vs 9417 shortest",
        ],
        15,
    ),
    RC(6.8, 1.6, 6.0, 4.6, "fdecea"),
    T(7.0, 1.7, 5.6, 0.5, "Did not hold up", 20, RED, True),
    B(
        7.0,
        2.3,
        5.6,
        3.8,
        [
            "Tornado: tie, and below published 0.874",
            "Ice: no win; gap closed -1.3%",
            "Wildfire WSTS 2021: DL loses",
            "Smoke CSI: persistence better",
            "IR, FLASH, HMS: no measurable help",
            "Tornado demo never blocks a road; May 20 tornado layer is empty",
        ],
        15,
    ),
    T(
        0.6,
        6.4,
        12.2,
        0.6,
        "Acceptance: 23 pass, 2 expected failures (AC-09 tornado and ice), AC-15 skipped in a clean clone",
        13,
        GREY,
    ),
)

slide(
    "Lessons and next steps",
    "What I learned. Baselines are not a formality: four of six models lost or tied somewhere. Evaluation design mattered "
    "more than architecture. A forecast only matters if it moves a decision. Calibration fixes probability quality, not ranking. "
    "Satellite channels added nothing here. And writing specs and tests first with an AI assistant filling in code made regressions "
    "catchable, but I had to review everything because tests do not prove the science. Next steps: close the tornado gap, fix the empty layer, "
    "retrain wildfire on out-of-time data, better ice and flood labels, and a multi-objective evacuation. Thank you.",
    T(0.7, 1.5, 5.9, 0.5, "What I learned", 20, ORANGE, True),
    B(
        0.7,
        2.1,
        5.9,
        4.6,
        [
            "Baselines are not a formality: four of six models lost or tied somewhere",
            "Evaluation design mattered more than architecture (NDWS win vs WSTS loss)",
            "A forecast matters only if it moves a decision",
            "Calibration fixes probability quality, not ranking",
            "AI-assisted spec-first work needs close review; tests do not prove the science",
        ],
        18,
    ),
    T(6.9, 1.5, 5.9, 0.5, "How I would improve it", 20, ORANGE, True),
    B(
        6.9,
        2.1,
        5.9,
        4.6,
        [
            "Tornado: understand the gap to 0.874; fix the empty May 20 layer; test a road-blocking policy",
            "Wildfire: fine-tune on out-of-time data; per-fire block bootstraps",
            "Ice and flood: better labels, more events, more routing scenarios",
            "Smoke: threshold-aware loss to compete on CSI",
            "Evacuation: close the gap to the lower bound; multi-objective",
        ],
        18,
    ),
)


# ---- PPTX ------------------------------------------------------------------------------------
def rgb(h):
    return RGBColor.from_string(h.upper())


def fit(name, w, h):
    iw, ih = Image.open(img(name)).size
    s = min(w / iw, h / ih)
    return iw * s, ih * s


def est_lines(text, w, size):
    cpl = max(1, int(w * 72 / (size * 0.52)))
    return sum(max(1, -(-len(p) // cpl)) for p in text.split("\n"))


def check_text(si, e, items, indent=0.0):
    w = e["w"] - indent - 0.2
    lines = sum(est_lines(i, w, e["size"]) for i in items)
    need = lines * e["size"] * 1.25 / 72 + 0.1 + (len(items) - 1) * (0.0 if e["k"] == "text" else 6 / 72)
    if need > e["h"] + 0.05:
        print(f"WARN slide {si} text may overflow: need {need:.2f} > {e['h']:.2f}: {items[0][:40]}")


def build_pptx():
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(W), Inches(HT)
    blank = prs.slide_layouts[6]
    for si, s in enumerate(S, 1):
        sl = prs.slides.add_slide(blank)
        bg = sl.background.fill
        bg.solid()
        bg.fore_color.rgb = rgb(NAVY if s["dark"] else "ffffff")
        if not s["dark"]:
            tb = sl.shapes.add_textbox(Inches(0.6), Inches(0.35), Inches(12.1), Inches(1.0))
            tf = tb.text_frame
            tf.word_wrap = True
            tf.vertical_anchor = MSO_ANCHOR.MIDDLE
            r = tf.paragraphs[0].add_run()
            r.text = s["title"]
            r.font.size, r.font.bold, r.font.name = Pt(28), True, FONT
            r.font.color.rgb = rgb(NAVY)
            bar = sl.shapes.add_shape(1, Inches(0.6), Inches(1.3), Inches(1.2), Inches(0.06))
            bar.fill.solid()
            bar.fill.fore_color.rgb = rgb(ORANGE)
            bar.line.fill.background()
            if len(s["title"]) > 62:
                print(f"WARN slide {si} long title")
        for e in s["els"]:
            k = e["k"]
            if k == "rect":
                sh = sl.shapes.add_shape(1, Inches(e["x"]), Inches(e["y"]), Inches(e["w"]), Inches(e["h"]))
                sh.fill.solid()
                sh.fill.fore_color.rgb = rgb(e["fill"])
                sh.line.fill.background()
                sh.shadow.inherit = False
            elif k in ("text", "bul"):
                tb = sl.shapes.add_textbox(Inches(e["x"]), Inches(e["y"]), Inches(e["w"]), Inches(e["h"]))
                tf = tb.text_frame
                tf.word_wrap = True
                tf.margin_left = tf.margin_right = Inches(0.05)
                items = e["paras"] if k == "text" else e["items"]
                check_text(si, e, items, 0.3 if k == "bul" else 0)
                for i, t in enumerate(items):
                    p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
                    r = p.add_run()
                    r.text = t
                    r.font.size, r.font.name = Pt(e["size"]), FONT
                    r.font.bold = bool(e.get("bold"))
                    r.font.color.rgb = rgb(e["color"])
                    if k == "text":
                        p.alignment = PP_ALIGN.LEFT
                        p.space_after = Pt(6) if len(items) > 1 else Pt(0)
                    else:
                        p.space_after = Pt(6)
                        pPr = p._p.get_or_add_pPr()
                        pPr.set("marL", "285750")
                        pPr.set("indent", "-285750")
                        bu = pPr.makeelement(qn("a:buChar"), {"char": "•"})
                        pPr.append(bu)
            elif k == "img":
                w, h = fit(e["name"], e["w"], e["h"])
                sl.shapes.add_picture(
                    img(e["name"]), Inches(e["x"] + (e["w"] - w) / 2), Inches(e["y"]), Inches(w), Inches(h)
                )
            elif k == "tbl":
                nr, nc = len(e["rows"]), len(e["cols"])
                gs = sl.shapes.add_table(
                    nr, nc, Inches(e["x"]), Inches(e["y"]), Inches(sum(e["cols"])), Inches(e["rh"] * nr)
                )
                tbl = gs.table
                tblPr = tbl._tbl.tblPr
                for a in ("bandRow", "firstRow"):
                    tblPr.set(a, "0")
                for j, cw in enumerate(e["cols"]):
                    tbl.columns[j].width = Inches(cw)
                for i, row in enumerate(e["rows"]):
                    tbl.rows[i].height = Inches(e["rh"])
                    for j, t in enumerate(row):
                        c = tbl.cell(i, j)
                        c.vertical_anchor = MSO_ANCHOR.MIDDLE
                        c.margin_left = c.margin_right = Inches(0.08)
                        c.fill.solid()
                        c.fill.fore_color.rgb = rgb(NAVY if i == 0 else ("ffffff" if i % 2 else LIGHT))
                        p = c.text_frame.paragraphs[0]
                        r = p.add_run()
                        r.text = t
                        r.font.size, r.font.name = Pt(e["size"]), FONT
                        col = "ffffff" if i == 0 else INK
                        bold = i == 0
                        for hi, hj, hc in e["hl"]:
                            if (hi, hj) == (i, j):
                                col, bold = hc, True
                        r.font.bold = bold
                        r.font.color.rgb = rgb(col)
                        if est_lines(t, e["cols"][j] - 0.16, e["size"]) > 1:
                            print(f"WARN slide {si} table cell wraps: {t[:40]}")
        sl.notes_slide.notes_text_frame.text = s["notes"]
    prs.save(out / "outrun.pptx")


# ---- PDF (same coordinates, HTML) -----------------------------------------------------------
def b64(name):
    return "data:image/png;base64," + base64.b64encode(Path(img(name)).read_bytes()).decode()


def build_pdf():
    pages = []
    for s in S:
        h = []
        if not s["dark"]:
            h.append(
                f'<div class="a" style="left:.6in;top:.35in;width:12.1in;height:1in;display:flex;align-items:center;'
                f'font-size:28pt;font-weight:bold;color:#{NAVY}">{H.escape(s["title"])}</div>'
            )
            h.append(
                f'<div class="a" style="left:.6in;top:1.3in;width:1.2in;height:.06in;background:#{ORANGE}"></div>'
            )
        for e in s["els"]:
            k = e["k"]
            pos = f"left:{e['x']}in;top:{e['y']}in;width:{e.get('w', 0)}in;height:{e.get('h', 0)}in;"
            if k == "rect":
                h.append(f'<div class="a" style="{pos}background:#{e["fill"]}"></div>')
            elif k == "text":
                ps = "".join(
                    f'<p style="margin:0 0 {6 if len(e["paras"]) > 1 else 0}pt">{H.escape(t)}</p>'
                    for t in e["paras"]
                )
                h.append(
                    f'<div class="a" style="{pos}font-size:{e["size"]}pt;color:#{e["color"]};'
                    f'font-weight:{"bold" if e["bold"] else "normal"};padding:0 .05in">{ps}</div>'
                )
            elif k == "bul":
                ps = "".join(f'<li style="margin:0 0 6pt">{H.escape(t)}</li>' for t in e["items"])
                h.append(
                    f'<ul class="a" style="{pos}font-size:{e["size"]}pt;color:#{e["color"]};margin:0;padding:0 0 0 .3in;'
                    f'box-sizing:border-box">{ps}</ul>'
                )
            elif k == "img":
                w, hh = fit(e["name"], e["w"], e["h"])
                h.append(
                    f'<img class="a" src="{b64(e["name"])}" style="left:{e["x"] + (e["w"] - w) / 2}in;top:{e["y"]}in;'
                    f'width:{w}in;height:{hh}in">'
                )
            elif k == "tbl":
                rows = ""
                for i, row in enumerate(e["rows"]):
                    tds = ""
                    for j, t in enumerate(row):
                        col, bold = ("ffffff" if i == 0 else INK), i == 0
                        for hi, hj, hc in e["hl"]:
                            if (hi, hj) == (i, j):
                                col, bold = hc, True
                        bg = NAVY if i == 0 else ("ffffff" if i % 2 else LIGHT)
                        tds += (
                            f'<td style="width:{e["cols"][j]}in;background:#{bg};color:#{col};'
                            f'font-weight:{"bold" if bold else "normal"}">{H.escape(t)}</td>'
                        )
                    rows += f'<tr style="height:{e["rh"]}in">{tds}</tr>'
                h.append(
                    f'<table class="a" style="left:{e["x"]}in;top:{e["y"]}in;font-size:{e["size"]}pt">{rows}</table>'
                )
        bg = NAVY if s["dark"] else "ffffff"
        pages.append(f'<section style="background:#{bg}">{"".join(h)}</section>')
    css = (
        f"@page{{size:{W}in {HT}in;margin:0}}body{{margin:0;font-family:{FONT},Helvetica,sans-serif}}"
        f"section{{position:relative;width:{W}in;height:{HT}in;overflow:hidden;page-break-after:always;break-after:page}}"
        ".a{position:absolute;box-sizing:border-box}table.a{border-collapse:collapse;table-layout:fixed}"
        "td{padding:0 .08in;border:0;vertical-align:middle;line-height:1.2;overflow:hidden}ul.a{line-height:1.2}p{line-height:1.2}"
    )
    with tempfile.TemporaryDirectory() as t:
        f = Path(t) / "s.html"
        f.write_text(f"<!doctype html><meta charset=utf-8><style>{css}</style>{''.join(pages)}")
        subprocess.run(
            [
                CHROME,
                "--headless=new",
                "--disable-gpu",
                "--no-pdf-header-footer",
                f"--print-to-pdf={out / 'outrun.pdf'}",
                f.as_uri(),
            ],
            check=True,
            capture_output=True,
            timeout=120,
        )


if __name__ == "__main__":
    out.mkdir(exist_ok=True)
    build_pptx()
    build_pdf()
    print(len(S), "slides")
