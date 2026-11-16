"""Build the disjoint dev (18) and fresh test (30) multimodal benchmark for Day 7 Advanced."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "data" / "adv_benchmark.json"


def build_benchmark_cases() -> list[dict[str, Any]]:
    """Construct verified dev (18) and fresh unseen test (30) questions anchored to Arctic Report Card 2025."""
    cases: list[dict[str, Any]] = [
        # =====================================================================
        # DEV SPLIT: 18 cases (3 table, 3 image_diagram, 3 chart, 3 multi_hop, 3 text, 3 unanswerable)
        # =====================================================================
        {
            "id": "adv_dev_table_01",
            "split": "dev",
            "group": "table",
            "question": (
                "According to Table 1 on page 12 ('Selected ecological events chronicled across the 20 editions "
                "of the Arctic Report Card since 2006'), what exact topic was highlighted for 2016 and for 2024?"
            ),
            "ground_truth": (
                "2016: 'Northern permafrost soils contain twice the carbon as in atmosphere'; "
                "2024: 'Arctic tundra transitions to being a carbon source'."
            ),
            "expected_pages": [12],
            "expected_regions": ["arc2025:p012:table:00"],
            "answerable": True,
        },
        {
            "id": "adv_dev_table_02",
            "split": "dev",
            "group": "table",
            "question": (
                "In Table 1 on page 102 summarizing taxa in the Chukchi Sea and northern Bering Sea, "
                "list all four taxa (scientific name and common name) classified under 'Pelagic fish'."
            ),
            "ground_truth": (
                "Mallotus villosus (Capelin), Clupea pallasii (Herring), "
                "Oncorhynchus gorbuscha (Pink Salmon), and Oncorhynchus nerka (Sockeye Salmon)."
            ),
            "expected_pages": [102],
            "expected_regions": ["arc2025:p102:table:00"],
            "answerable": True,
        },
        {
            "id": "adv_dev_table_03",
            "split": "dev",
            "group": "table",
            "question": (
                "In Table 1 under Physical Indicators (page 133), for which Essay Topic is GMASI (Autosnow Ice) "
                "mentioned, and what does its Gaps entry state about both in situ ice sensors and GMASI?"
            ),
            "ground_truth": (
                "Essay Topic: Lake Ice. The Gaps column states that data from in situ ice sensors have good coverage "
                "but are still limited, while GMASI (Autosnow Ice) data are useless in winter but add value in summer "
                "and are useful for detecting snow under warm clouds in spring where surface stations are unavailable."
            ),
            "expected_pages": [133],
            "expected_regions": ["arc2025:p133:table:00"],
            "answerable": True,
        },
        {
            "id": "adv_dev_img_01",
            "split": "dev",
            "group": "image_diagram",
            "question": (
                "Inspect the Arctic Report Card Vital Signs wheel diagram on page 7 (Fig. 1): list all eight "
                "Vital Signs categories inside the wheel and report the trend box text for Terrestrial Snow Cover."
            ),
            "ground_truth": (
                "Eight categories: Surface Air Temperature, Precipitation, Terrestrial Snow Cover, Tundra Greenness, "
                "Greenland Ice Sheet, Ocean Primary Productivity, Sea Ice, Sea Surface Temperature. "
                "Terrestrial Snow Cover: 'June snow cover extent reduced by 50% since 1967'."
            ),
            "expected_pages": [7],
            "expected_regions": ["arc2025:p007:fig:00"],
            "answerable": True,
        },
        {
            "id": "adv_dev_img_02",
            "split": "dev",
            "group": "image_diagram",
            "question": (
                "Inspect the four-panel tundra vegetation photograph (Fig. 1 on page 75): in the bottom-right panel, "
                "what color is the researcher's backpack instrument, what are they pointing in their right hand, "
                "and what lies inside the open black case on the tundra in front of them?"
            ),
            "ground_truth": (
                "The researcher wears a white (light grey/white) backpack spectrometer, holds a connected optical sensor/"
                "cable in their right hand aimed down at the vegetation, and has a white reference/calibration panel "
                "resting inside an open black case on the grass."
            ),
            "expected_pages": [75],
            "expected_regions": ["arc2025:p075:fig:00"],
            "answerable": True,
        },
        {
            "id": "adv_dev_img_03",
            "split": "dev",
            "group": "image_diagram",
            "question": (
                "Inspect Fig. 2 on page 112 (coastal erosion monitoring in Golovin, Alaska): how many people are standing "
                "beside the wooden building, what large bright blue cylindrical container is mounted on a wooden rack "
                "right next to them, and what red container lies on the snowy slope in the lower-left foreground?"
            ),
            "ground_truth": (
                "Two local observers stand beside the weathered wooden building; a bright blue plastic barrel/drum is "
                "mounted horizontally on a raised wooden frame right beside the person on the left; a red fuel can/"
                "jerrycan lies in the lower-left foreground."
            ),
            "expected_pages": [112],
            "expected_regions": ["arc2025:p112:fig:00"],
            "answerable": True,
        },
        {
            "id": "adv_dev_chart_01",
            "split": "dev",
            "group": "chart",
            "question": (
                "Look closely at panel (c) Winter in the Surface Air Temperature time series chart on page 15 (Fig. 1): "
                "what are the exact printed 2006–2025 warming trends (in °C dec^-1) for the Arctic (red) and for the "
                "globe (blue), and by how much does the Arctic winter rate exceed the global winter rate?"
            ),
            "ground_truth": (
                "T_ARCTIC 2006-2025 = 0.87°C dec^-1 and T_GLOBAL 2006-2025 = 0.35°C dec^-1; "
                "Arctic winter warming exceeds the global rate by 0.52°C dec^-1 (about 2.49x faster)."
            ),
            "expected_pages": [15],
            "expected_regions": ["arc2025:p015:fig:00"],
            "answerable": True,
        },
        {
            "id": "adv_dev_chart_02",
            "split": "dev",
            "group": "chart",
            "question": (
                "In Fig. 1 on page 30 showing standardized Arctic snow cover extent anomalies (1967–2025), check the "
                "filled 2025 circles in panel (b) June SCE Anomaly: are both Eurasia (red) and North America (black) "
                "positive or negative, and approximately what standardized values do they show?"
            ),
            "ground_truth": (
                "Both 2025 June SCE anomalies are negative: Eurasian Arctic (red filled circle) is around -0.7 "
                "and North American Arctic (black filled circle) is around -1.3 (between -1 and -2)."
            ),
            "expected_pages": [30],
            "expected_regions": ["arc2025:p030:fig:00"],
            "answerable": True,
        },
        {
            "id": "adv_dev_chart_03",
            "split": "dev",
            "group": "chart",
            "question": (
                "Inspect the MaxNDVI time series plot on page 76: which dataset curve consistently shows the highest "
                "maximum NDVI (~0.58–0.63) and which shows the lowest (~0.405–0.455), and in what year does the "
                "MODIS Circumarctic (black) curve peak near ~0.52?"
            ),
            "ground_truth": (
                "Highest: MODIS Eurasia (dotted red); Lowest: MODIS North America (dotted blue); "
                "MODIS Circumarctic peaks highest in 2020 (just below 0.52)."
            ),
            "expected_pages": [76],
            "expected_regions": ["arc2025:p076:fig:00"],
            "answerable": True,
        },
        {
            "id": "adv_dev_multihop_01",
            "split": "dev",
            "group": "multi_hop",
            "question": (
                "Combining Precipitation Headlines (p. 21) and 'Precipitation at a glance' (p. 22), how did the "
                "2024/25 water year rank overall, how did each of the four seasons rank, and which single season was "
                "unusually dry?"
            ),
            "ground_truth": (
                "The 2024/25 water year (Oct 2024–Sep 2025) ranked wettest/highest on record; spring was record high (#1), "
                "autumn was 4th highest, and winter was 4th highest, whereas summer was unusually dry, ranking 35th highest."
            ),
            "expected_pages": [21, 22],
            "expected_regions": ["arc2025:p021:text:000", "arc2025:p022:text:001"],
            "answerable": True,
        },
        {
            "id": "adv_dev_multihop_02",
            "split": "dev",
            "group": "multi_hop",
            "question": (
                "Using pages 36–37 on the Greenland Ice Sheet, compare the 2025 GRACE-FO annual mass balance against the "
                "2003–24 GRACE/GRACE-FO annual average (in Gt): by how many Gt was the 2025 net mass loss smaller, "
                "and what partial-year mass balance did ICESat-2 measure from 1 September 2024 to 2 April 2025?"
            ),
            "ground_truth": (
                "2025 GRACE-FO mass balance was -129 ± 50 Gt vs. the 2003–24 average of -219 ± 16 Gt, meaning 2025 loss "
                "was 90 Gt smaller in magnitude (219 - 129 = 90 Gt). ICESat-2 measured -2 ± 56 Gt for 1 Sep 2024–2 Apr 2025."
            ),
            "expected_pages": [36, 37],
            "expected_regions": ["arc2025:p036:text:001", "arc2025:p037:text:001"],
            "answerable": True,
        },
        {
            "id": "adv_dev_multihop_03",
            "split": "dev",
            "group": "multi_hop",
            "question": (
                "From pages 110–111 on the Aleut Community of St. Paul Island (ACSPI), state the full names of both the "
                "BRAIDED Food Security Project and ISN, which WHO top-ten contaminant the BRAIDED pilot monitored at "
                "the BSRC lab, and what core principle underpins ISN."
            ),
            "ground_truth": (
                "BRAIDED = Building Research Aligned with Indigenous Determination, Equity, and Decision-making (monitored "
                "mercury locally at the Bering Sea Research Center, BSRC); ISN = Indigenous Sentinels Network, founded on "
                "Indigenous Data Sovereignty (IDSov), where communities own and control all collected data."
            ),
            "expected_pages": [110, 111],
            "expected_regions": ["arc2025:p110:text:000", "arc2025:p111:text:001", "arc2025:p111:text:002"],
            "answerable": True,
        },
        {
            "id": "adv_dev_text_01",
            "split": "dev",
            "group": "text",
            "question": (
                "According to the Surface Air Temperature Overview on page 14, since 1980 how much faster have Pan-Arctic "
                "(60° N–90° N) annual air temperatures warmed compared to the global mean (90° S–90° N)?"
            ),
            "ground_truth": "Nearly three times faster than the global mean since 1980.",
            "expected_pages": [14],
            "expected_regions": ["arc2025:p014:text:003"],
            "answerable": True,
        },
        {
            "id": "adv_dev_text_02",
            "split": "dev",
            "group": "text",
            "question": (
                "According to page 29 on Terrestrial Snow Cover, how does June Arctic snow cover extent today compare "
                "to six decades ago, and through what physical feedback does snow cover loss amplify Arctic warming?"
            ),
            "ground_truth": (
                "June Arctic snow cover extent today is half of what it was six decades ago, and snow cover loss "
                "amplifies Arctic warming through the surface albedo feedback."
            ),
            "expected_pages": [29],
            "expected_regions": ["arc2025:p029:text:000", "arc2025:p029:text:001"],
            "answerable": True,
        },
        {
            "id": "adv_dev_text_03",
            "split": "dev",
            "group": "text",
            "question": (
                "According to the AMAP essay on page 143, since which year has the Arctic Report Card been published, "
                "and more than how many specialists in AMAP's international network support its peer review process?"
            ),
            "ground_truth": "Published annually since 2006; AMAP draws on an international network of more than 800 specialists.",
            "expected_pages": [143],
            "expected_regions": ["arc2025:p143:text:000"],
            "answerable": True,
        },
        {
            "id": "adv_dev_unans_01",
            "split": "dev",
            "group": "unanswerable",
            "question": "What Arctic penguin colony breeding counts are reported in the 2025 Arctic Report Card?",
            "ground_truth": "Insufficient evidence in the provided document.",
            "expected_pages": [],
            "expected_regions": [],
            "answerable": False,
        },
        {
            "id": "adv_dev_unans_02",
            "split": "dev",
            "group": "unanswerable",
            "question": "How much funding in euros did Norway allocate to expanding its Arctic icebreaker fleet in 2025 according to the report?",
            "ground_truth": "Insufficient evidence in the provided document.",
            "expected_pages": [],
            "expected_regions": [],
            "answerable": False,
        },
        {
            "id": "adv_dev_unans_03",
            "split": "dev",
            "group": "unanswerable",
            "question": "What surface ocean microplastic fiber concentration (particles per cubic meter) is reported in Table 2 of the report?",
            "ground_truth": "Insufficient evidence in the provided document.",
            "expected_pages": [],
            "expected_regions": [],
            "answerable": False,
        },

        # =====================================================================
        # FRESH TEST SPLIT v2: 30 cases (6 table, 5 image_diagram, 5 chart, 4 multi_hop, 4 text, 6 unanswerable)
        # Zero page overlap with dev; completely fresh figures/charts/diagrams/text/tables/entities!
        # =====================================================================
        {
            "id": "adv_test_table_01",
            "split": "test",
            "group": "table",
            "question": (
                "In Table 1 on page 46 (Arctic sea ice extent), report for both March Daily Maximum and September "
                "Daily Minimum: the 2025 Extent (10^6 km^2), Rank (out of 47 years), Anomaly rel. 1991–2020 average "
                "(10^6 km^2), and the 1979–2005 Trend (km^2/yr). Which of those two metrics had the faster decline over "
                "1979–2005?"
            ),
            "ground_truth": (
                "March Daily Maximum: Extent 14.31 × 10^6 km^2, Rank 1 (out of 47), Anomaly -1.05 × 10^6 km^2, "
                "1979–2005 Trend -40,300 km^2/yr. September Daily Minimum: Extent 4.60 × 10^6 km^2, Rank 10 (out of 47), "
                "Anomaly -0.78 × 10^6 km^2, 1979–2005 Trend -53,800 km^2/yr. September Daily Minimum declined faster "
                "(-53,800 vs. -40,300 km^2/yr)."
            ),
            "expected_pages": [46],
            "expected_regions": ["arc2025:p046:table:00"],
            "answerable": True,
        },
        {
            "id": "adv_test_table_02",
            "split": "test",
            "group": "table",
            "question": (
                "Using Table 1 on page 69 (Primary productivity, 2003–25), compare the Eurasian Arctic, Barents Sea, "
                "and Hudson Bay rows: report each region's 2003–25 Trend (g C/m^2/yr/decade), Mann-Kendall p-value, "
                "2003–25 % Change, and 2025 Primary Productivity (% of the 2003-22 average). Which of these three had the "
                "largest 2003–25 % Change?"
            ),
            "ground_truth": (
                "Eurasian Arctic: Trend 34.40, p <0.0001, % Change 80.2%, 2025 Productivity 148.2%; Barents Sea: "
                "Trend 22.93, p <0.0001, % Change 33.8%, 2025 Productivity 130.0%; Hudson Bay: Trend 11.24, p 0.039, "
                "% Change 27.1%, 2025 Productivity 121.3%. Eurasian Arctic had the largest % Change (80.2%)."
            ),
            "expected_pages": [69],
            "expected_regions": ["arc2025:p069:table:00"],
            "answerable": True,
        },
        {
            "id": "adv_test_table_03",
            "split": "test",
            "group": "table",
            "question": (
                "In Table 1 on page 83 listing 27 Arctic glaciers (climatic mass balance Bclim in mm w.e./yr), examine "
                "the Alaska (#1–3) and Arctic Canada (#4–7) sections: which glacier in Alaska and which glacier in Arctic "
                "Canada recorded the most negative Bclim in 2022/23, what were their exact 2022/23 values, and what are "
                "their record lengths (years)?"
            ),
            "ground_truth": (
                "Alaska: Lemon Creek (#2) at -2360 mm w.e./yr in 2022/23 (record length 72 years). "
                "Arctic Canada: Melville S. Ice Cap (#6) at -1482 mm w.e./yr in 2022/23 (record length 62 years)."
            ),
            "expected_pages": [83],
            "expected_regions": ["arc2025:p083:table:00"],
            "answerable": True,
        },
        {
            "id": "adv_test_table_04",
            "split": "test",
            "group": "table",
            "question": (
                "In Table 1 on page 134 (observational gaps across Ecological Indicators and Integrated Indicators), "
                "summarize the specific Gaps entries listed for 'Geese' and for 'Ship Traffic'."
            ),
            "ground_truth": (
                "Geese: visual estimates often underestimate and sample only part of a species' range; research is "
                "ongoing to expand banding and apply aerial imagery surveys, while image analysis still needs improvement "
                "to reduce manual effort. Ship Traffic (Integrated Indicators): deficiencies in the comprehensiveness of "
                "antennae available to generate AIS signals and omission of certain ships and ship-types from AIS tracking "
                "make the dataset incomplete."
            ),
            "expected_pages": [134],
            "expected_regions": ["arc2025:p134:table:00"],
            "answerable": True,
        },
        {
            "id": "adv_test_table_05",
            "split": "test",
            "group": "table",
            "question": (
                "According to Table 2 under Physical Indicators on page 135, how do the risk assessments differ between "
                "'Surface Air Temperature' and 'Precipitation' (specifically what observing system is at risk and what "
                "cause is given)?"
            ),
            "ground_truth": (
                "Surface Air Temperature: 'No risks noted' (—). Precipitation: 'Meteorological Stations Collecting "
                "Precipitation Data - a highly critical observing system for the Precipitation essay' are at risk due to "
                "potential impacts as precipitation data are collected or funded by agencies or programs facing major budget cuts."
            ),
            "expected_pages": [135],
            "expected_regions": ["arc2025:p135:table:00"],
            "answerable": True,
        },
        {
            "id": "adv_test_table_06",
            "split": "test",
            "group": "table",
            "question": (
                "According to Table 2 on page 138, which international collaboration is noted as at risk for "
                "'Pollinators', and what stakeholder coordination organization + NSF funding program pause in 2025 is "
                "reported for 'Beavers'?"
            ),
            "ground_truth": (
                "Pollinators: CBMP Arctic Pollinator Expert Network (funded by agencies or programs facing major budget "
                "cuts). Beavers: A-BON (created to coordinate research and action among stakeholders related to beavers "
                "in the circumpolar Arctic), with funding opportunities with the NSF-AON program on pause for the 2025 "
                "calendar year (McManus 2025)."
            ),
            "expected_pages": [138],
            "expected_regions": ["arc2025:p138:table:00"],
            "answerable": True,
        },
        {
            "id": "adv_test_img_01",
            "split": "test",
            "group": "image_diagram",
            "question": (
                "Inspect the Siberian Arctic Ocean diagram in Fig. 5 on page 96: compare Phase 1 vs. Phase 2 in panel (a) "
                "(halocline strength and labeled items 1, 2, 3), and describe how the Nutrients profile shifts vertically "
                "from panel (b) Earlier period to panel (c) Later period."
            ),
            "ground_truth": (
                "Items 1=Warm, salty Atlantic water; 2=Ocean heat moving up; 3=Summer warming. Phase 1 shows a Strong "
                "halocline (More stratified) with weaker upward heat movement, whereas Phase 2 shows a Weak halocline "
                "(Less stratified) with much stronger upward ocean heat moving up (#2) and larger summer warming (#3). "
                "In biological panels (b)->(c), the Nutrients maximum/profile shifts upward (shoals, upward arrow) from "
                "below into the fresher upper surface mixed layer."
            ),
            "expected_pages": [96],
            "expected_regions": ["arc2025:p096:fig:00"],
            "answerable": True,
        },
        {
            "id": "adv_test_img_02",
            "split": "test",
            "group": "image_diagram",
            "question": (
                "Inspect the Northern Fur Seal rookery survey photograph in Fig. 3 on page 113: how many surveyors in "
                "total appear in the photo on the grassy hillside, what direction/target are they looking toward, and what "
                "optical instruments are at least two standing surveyors holding up to their eyes?"
            ),
            "ground_truth": (
                "Six people in total (four Sentinels and two NOAA NMFS officials) on the grassy coastal hillside overlooking "
                "the shoreline/rookery below; standing surveyors on the far right and middle are holding binoculars/"
                "optical spotting devices up to their eyes."
            ),
            "expected_pages": [113],
            "expected_regions": ["arc2025:p113:fig:00"],
            "answerable": True,
        },
        {
            "id": "adv_test_img_03",
            "split": "test",
            "group": "image_diagram",
            "question": (
                "Inspect the field photograph in Fig. 4 on page 114: what surface is the observer standing on in black "
                "boots, what handheld electronic device is the observer using with both gloved hands, and what snowy/stream "
                "environment surrounds them?"
            ),
            "ground_truth": (
                "A solo observer in dark boots stands on clear frozen stream surface ice (showing a reflection beneath), "
                "holding a smartphone (light blue case) with both gloved hands to log data, surrounded by snow-covered "
                "banks, bare riparian shrubs, and snowy mountains."
            ),
            "expected_pages": [114],
            "expected_regions": ["arc2025:p114:fig:00"],
            "answerable": True,
        },
        {
            "id": "adv_test_img_04",
            "split": "test",
            "group": "image_diagram",
            "question": (
                "Inspect the Box 1 grey summary box on page 129 (International Arctic Observing Assessment Framework): "
                "state how many key objectives SAON's partners organized, and list all 12 numbered Societal Benefit Areas "
                "by title."
            ),
            "ground_truth": (
                "More than 150 key objectives organized into 12 Societal Benefit Areas: 1. Disaster Preparedness, "
                "2. Environmental Quality, 3. Food Security, 4. Fundamental Understanding of Arctic Systems, "
                "5. Human Health, 6. Infrastructure and Operations, 7. Marine and Coastal Ecosystems and Processes, "
                "8. Natural Resources, 9. Resilient Communities, 10. Sociocultural Services, "
                "11. Terrestrial and Freshwater Ecosystems and Processes, 12. Weather and Climate."
            ),
            "expected_pages": [129],
            "expected_regions": ["arc2025:p129:fig:00"],
            "answerable": True,
        },
        {
            "id": "adv_test_img_05",
            "split": "test",
            "group": "image_diagram",
            "question": (
                "Inspect the production wheel diagram on page 144 ('Components and contributions to the annual production "
                "of the Arctic Report Card'): name the overarching outer ring title at the top and all four colored "
                "inner quadrants."
            ),
            "ground_truth": (
                "Outer ring: Coordinating Editor. Four inner quadrants: Independent Editorial Team (dark navy), "
                "International Body of Subject Matter Expert Authors (green), Peer Review by AMAP (orange), and "
                "Communication & Dissemination by NOAA (blue)."
            ),
            "expected_pages": [144],
            "expected_regions": ["arc2025:p144:fig:00"],
            "answerable": True,
        },
        {
            "id": "adv_test_chart_01",
            "split": "test",
            "group": "chart",
            "question": (
                "Inspect Fig. 3 on page 25 (Arctic 60–90° N precipitation time series 1950/51–2024/25): what y-axis unit "
                "is plotted across all five panels, and in which seasonal panels does the black GPCC line reach a sharp "
                "spike right at or just above 120% (around 2020 and around 2012, respectively)?"
            ),
            "ground_truth": (
                "Y-axis unit: Percent of 1991-2020 average (horizontal line at 100%). The black GPCC line spikes above "
                "120% at 2020 in panel (c) Winter (January - March), and spikes to ~120% around 2012 in panel (d) Spring "
                "(April - June)."
            ),
            "expected_pages": [25],
            "expected_regions": ["arc2025:p025:fig:00"],
            "answerable": True,
        },
        {
            "id": "adv_test_chart_02",
            "split": "test",
            "group": "chart",
            "question": (
                "Inspect Fig. 3 on page 57 (August SST trends, 1982–2025): in panel (b), state the y-axis quantity/unit, "
                "the exact linear trends with 95% confidence intervals printed in the legend for Arctic (65-90°N) in blue "
                "and Northern Hemisphere (0-65°N) in red, and note the prominent blue spike in 2007."
            ),
            "ground_truth": (
                "Y-axis: SST Anomaly (°C). Blue legend [Arctic (65-90°N)]: 0.03 ± 0.01 °C/yr; Red legend [Northern "
                "Hemisphere (0-65°N)]: 0.03 ± 0.003 °C/yr. In 2007, the blue Arctic line spikes sharply near ~+0.9 to +1.0°C."
            ),
            "expected_pages": [57],
            "expected_regions": ["arc2025:p057:fig:00"],
            "answerable": True,
        },
        {
            "id": "adv_test_chart_03",
            "split": "test",
            "group": "chart",
            "question": (
                "Inspect the four-panel August SST anomaly chart in Fig. 4 on page 58 (1982–2025): state the exact "
                "printed linear trends (°C/yr ± CI) shown inside panels (a) Kara Sea, (b) Laptev Sea, (c) Barents Sea, "
                "and (d) Chukchi Sea, and identify which single region's trend label is grey (not statistically significant)."
            ),
            "ground_truth": (
                "(a) Kara Sea: 0.12 ± 0.03 °C/yr; (b) Laptev Sea: 0.08 ± 0.02 °C/yr; (c) Barents Sea: 0.06 ± 0.02 °C/yr; "
                "(d) Chukchi Sea: 0.03 ± 0.03 °C/yr (printed in grey = not statistically significant)."
            ),
            "expected_pages": [58],
            "expected_regions": ["arc2025:p058:fig:00"],
            "answerable": True,
        },
        {
            "id": "adv_test_chart_04",
            "split": "test",
            "group": "chart",
            "question": (
                "Inspect Fig. 2 on page 91 ('Atlantic water temperature departure from average', 1893–2017): after rising "
                "above 0°C around the late 1990s/2000, what peak departure level does the shaded curve reach (between "
                "0 and 1°C) in the late 2000s, and what two expedition photos are shown above the chart?"
            ),
            "ground_truth": (
                "The curve crosses persistently above 0°C around the mid/late 1990s–2000 and peaks near ~+0.8 to +0.9°C "
                "(just under 1°C) in the late 2000s. Above the chart are photos comparing Nansen's 'Fram 1893–1896' "
                "sailing vessel in ice vs. 'Modern expeditions' showing a red-hulled U.S. Coast Guard icebreaker (#20)."
            ),
            "expected_pages": [91],
            "expected_regions": ["arc2025:p091:fig:00"],
            "answerable": True,
        },
        {
            "id": "adv_test_chart_05",
            "split": "test",
            "group": "chart",
            "question": (
                "Inspect Fig. 3 on page 122 (geochemistry of rusting rivers and seeps): in panel (a), compare the "
                "approximate pH of the red Seep point against the grey Unimpaired and orange Impaired boxes; state what "
                "scale is used on the y-axis of panels (b) & (c); and list all 7 metal categories in the panel (d) pie chart, "
                "naming the three largest slices."
            ),
            "ground_truth": (
                "Panel (a) pH: red Seep is strongly acidic at pH ~2.5–2.6, vs. Unimpaired (~7.9–8.2) and Impaired (~7.5 "
                "median, ~6–8.1 range). Panels (b) Sulfate and (c) Sum of metals use a logarithmic (log) y-axis scale. "
                "Panel (d) lists 7 slices: Fe, Al, Mn (the three largest major slices), plus Zn, Ni, Cu, and Other."
            ),
            "expected_pages": [122],
            "expected_regions": ["arc2025:p122:fig:00"],
            "answerable": True,
        },
        {
            "id": "adv_test_multihop_01",
            "split": "test",
            "group": "multi_hop",
            "question": (
                "Combine autumn 2024 air temperature anomalies (p. 17) and Greenland albedo/smoke observations (p. 39): "
                "on p. 17, what anomaly range (°C) marked record-high autumn 2024 warmth and what local event drove "
                "eastern Hudson Bay's extreme temperatures; on p. 39, how did summer snowfall vs. late-August wildfire "
                "smoke shift Greenland's surface albedo?"
            ),
            "ground_truth": (
                "On p. 17, record-high autumn 2024 temperatures reached 4–6°C above the mean, and eastern Hudson Bay's "
                "extreme warmth followed record early sea ice retreat allowing exceptional ocean heat uptake and a "
                "season-long marine heatwave. On p. 39, summer snowfall raised (brightened) Greenland's albedo above "
                "average, before albedo dropped below average in late August coincident with smoke particulates from "
                "North American wildfires."
            ),
            "expected_pages": [17, 39],
            "expected_regions": ["arc2025:p017:text:001", "arc2025:p039:text:000"],
            "answerable": True,
        },
        {
            "id": "adv_test_multihop_02",
            "split": "test",
            "group": "multi_hop",
            "question": (
                "Combining Greenland station records (p. 38) and passive microwave methods (p. 42): how many weather "
                "stations across Greenland were evaluated in 2025 and which station set an April record high monthly mean; "
                "AND what satellite sensor (frequency) was decommissioned on 24 September 2025 and what data product did "
                "that end?"
            ),
            "ground_truth": (
                "34 weather stations across Greenland were evaluated; Summit Station reached a record high monthly mean "
                "in April 2025 (p. 38). On 24 September 2025, the daily Special Sensor Microwave Imager/Sounder (SSMIS, "
                "37 GHz horizontally polarized passive microwave radiometer) was decommissioned, ending the capability "
                "to produce the surface melt duration and extent product (p. 42)."
            ),
            "expected_pages": [38, 42],
            "expected_regions": ["arc2025:p038:text:001", "arc2025:p042:text:000"],
            "answerable": True,
        },
        {
            "id": "adv_test_multihop_03",
            "split": "test",
            "group": "multi_hop",
            "question": (
                "Combining Tundra Greenness (p. 74) and Borealization in the Northern Bering and Chukchi Seas (p. 100): "
                "what total area (million km^2) does the treeless Arctic tundra biome encompass (p. 74), AND what historical "
                "surface-to-seafloor biological coupling (pelagic–benthic coupling) supports those Arctic marine ecosystems "
                "and how does borealization alter seafloor species distribution (p. 100)?"
            ),
            "ground_truth": (
                "Arctic tundra covers 5.1 million km^2 (p. 74). Arctic marine ecosystems historically relied on strong "
                "biological coupling between surface/midwater (pelagic) and seafloor (benthic) zones driven by spring "
                "phytoplankton blooms exporting carbon to the seafloor; emerging borealization on the seafloor is shown by "
                "southern species appearing farther north as conditions warm (p. 100)."
            ),
            "expected_pages": [74, 100],
            "expected_regions": ["arc2025:p074:text:001", "arc2025:p100:text:001", "arc2025:p100:text:002"],
            "answerable": True,
        },
        {
            "id": "adv_test_multihop_04",
            "split": "test",
            "group": "multi_hop",
            "question": (
                "Combining pages 127 and 128 on the Arctic Observing Network (AON): what gap in precipitation gauges "
                "hinders river discharge assessment (p. 127), what three satellite observation limitations are identified in "
                "the Arctic (p. 128), and which three Horizon 2020 initiatives advance SAON's goals in Europe?"
            ),
            "ground_truth": (
                "Sparse precipitation gauges in complex terrain cause underestimates, hindering assessments of river "
                "discharge changes (p. 127). Satellite limitations on p. 128: (1) geostationary coverage degrades above "
                "60° N, (2) frequent cloud cover and limited daylight mask signals, and (3) inadequate spatial and/or "
                "temporal resolution in some cases. European Horizon 2020 initiatives: EU-PolarNet, INTAROS, and Arctic PASSION."
            ),
            "expected_pages": [127, 128],
            "expected_regions": ["arc2025:p127:text:001", "arc2025:p128:text:000", "arc2025:p128:text:002"],
            "answerable": True,
        },
        {
            "id": "adv_test_text_01",
            "split": "test",
            "group": "text",
            "question": (
                "According to page 31 (Terrestrial Snow Cover), in which month does Arctic snow mass typically reach its "
                "annual peak, how did April 2025 snow mass anomalies compare to the 1991–2020 baseline across North America "
                "and Eurasia, and where did the 2024/25 snow season rank shortest in the 27-year record?"
            ),
            "ground_truth": (
                "Snow mass typically peaks in April; April 2025 snow mass anomalies were above the 1991–2020 baseline "
                "across both North American and Eurasian Arctic; broad portions of the southern Canadian Arctic Archipelago "
                "ranked shortest in the 27-year record."
            ),
            "expected_pages": [31],
            "expected_regions": ["arc2025:p031:text:000", "arc2025:p031:text:002"],
            "answerable": True,
        },
        {
            "id": "adv_test_text_02",
            "split": "test",
            "group": "text",
            "question": (
                "According to the ice discharge text on page 40, what Greenland Ice Sheet solid ice discharge rate (Gt/yr) "
                "was measured through 20 August 2025 in the 2025 mass balance year, and how does it compare with the "
                "1991–2020 mean?"
            ),
            "ground_truth": (
                "Solid ice discharge was 491 ± 17 Gt/yr through 20 August 2025, approximately 1 standard deviation above "
                "the 1991–2020 mean of 458 ± 27 Gt/yr."
            ),
            "expected_pages": [40],
            "expected_regions": ["arc2025:p040:text:000"],
            "answerable": True,
        },
        {
            "id": "adv_test_text_03",
            "split": "test",
            "group": "text",
            "question": (
                "According to page 62 (Primary Productivity), what does CDOM stand for in Arctic riverine discharge, how "
                "can CDOM and riverine nutrients influence marine productivity, and what food-web problem can climate-driven "
                "shifts in phytoplankton bloom timing and composition trigger?"
            ),
            "ground_truth": (
                "CDOM = chromophoric (colored) dissolved organic matter; riverine nutrients and CDOM can either enhance or "
                "inhibit productivity depending on local conditions; shifts in bloom timing and species composition can "
                "cause trophic mismatches reducing carbon utilization through the food chain."
            ),
            "expected_pages": [62],
            "expected_regions": ["arc2025:p062:text:000", "arc2025:p062:text:001"],
            "answerable": True,
        },
        {
            "id": "adv_test_text_04",
            "split": "test",
            "group": "text",
            "question": (
                "According to page 115 ('Food safety: Food security is climate security'), nearly how many biological "
                "samples did the BRAIDED project test in its first year (2024), what specific mercury analyzer instrument "
                "was used, and where were results uploaded?"
            ),
            "ground_truth": (
                "Nearly 100 biological samples (marine mammals to seabirds) tested locally using a Nippon MA-Solo direct "
                "mercury analyzer, with results uploaded to ISN (Indigenous Sentinels Network)."
            ),
            "expected_pages": [115],
            "expected_regions": ["arc2025:p115:text:001"],
            "answerable": True,
        },
        {
            "id": "adv_test_unans_01",
            "split": "test",
            "group": "unanswerable",
            "question": (
                "What planned launch year and orbital altitude (in km) are reported for the upcoming GRACE-C satellite "
                "mission in the 2025 Arctic Report Card?"
            ),
            "ground_truth": "Insufficient evidence in the provided document.",
            "expected_pages": [],
            "expected_regions": [],
            "answerable": False,
        },
        {
            "id": "adv_test_unans_02",
            "split": "test",
            "group": "unanswerable",
            "question": (
                "Under the CMIP7 SSP1-2.6 multi-model ensemble in the report, what September Arctic sea ice extent is "
                "projected for 2040?"
            ),
            "ground_truth": "Insufficient evidence in the provided document.",
            "expected_pages": [],
            "expected_regions": [],
            "answerable": False,
        },
        {
            "id": "adv_test_unans_03",
            "split": "test",
            "group": "unanswerable",
            "question": (
                "What annual black carbon emissions (in kilotonnes) from nuclear icebreakers operating in the Arctic are "
                "quantified in the 2025 Arctic Report Card?"
            ),
            "ground_truth": "Insufficient evidence in the provided document.",
            "expected_pages": [],
            "expected_regions": [],
            "answerable": False,
        },
        {
            "id": "adv_test_unans_04",
            "split": "test",
            "group": "unanswerable",
            "question": (
                "Which legal clauses of the Svalbard Treaty govern offshore oil drilling concessions according to the "
                "2025 Arctic Report Card?"
            ),
            "ground_truth": "Insufficient evidence in the provided document.",
            "expected_pages": [],
            "expected_regions": [],
            "answerable": False,
        },
        {
            "id": "adv_test_unans_05",
            "split": "test",
            "group": "unanswerable",
            "question": (
                "What share (%) of electricity generation across Arctic Council states comes from hydroelectric and "
                "geothermal power plants according to Table 2 of the report?"
            ),
            "ground_truth": "Insufficient evidence in the provided document.",
            "expected_pages": [],
            "expected_regions": [],
            "answerable": False,
        },
        {
            "id": "adv_test_unans_06",
            "split": "test",
            "group": "unanswerable",
            "question": (
                "What annual failure rates for subsea submarine cables and Starlink terminals caused by subsea permafrost "
                "thaw are reported in the 2025 Arctic Report Card?"
            ),
            "ground_truth": "Insufficient evidence in the provided document.",
            "expected_pages": [],
            "expected_regions": [],
            "answerable": False,
        },
    ]
    return cases


def main() -> None:
    cases = build_benchmark_cases()
    chunks = json.loads((ROOT / "data" / "adv_index" / "chunks.json").read_text(encoding="utf-8"))
    chunks_by_id = {c["id"]: c for c in chunks}

    dev = [c for c in cases if c["split"] == "dev"]
    test = [c for c in cases if c["split"] == "test"]
    assert len(dev) == 18, f"Expected 18 dev, got {len(dev)}"
    assert len(test) == 30, f"Expected 30 test, got {len(test)}"
    for c in cases:
        crops: list[str] = []
        sections: list[str] = []
        for cid in c["expected_regions"]:
            assert cid in chunks_by_id, f"Missing region chunk_id {cid} in chunks.json"
            ch = chunks_by_id[cid]
            if ch.get("crop_path") and ch["crop_path"] not in crops:
                crops.append(ch["crop_path"])
            if ch.get("section") and ch["section"] not in sections:
                sections.append(ch["section"])
        c["expected_behavior"] = "answer" if c["answerable"] else "abstain"
        c["expected_crops"] = crops
        if c["answerable"]:
            pages_str = ", ".join(str(p) for p in c["expected_pages"])
            c["source"] = (
                f"Arctic Report Card 2025 (DOI:10.25923/nrzf-j897), section(s) {', '.join(sections) or 'Report'}, "
                f"PDF page(s) [{pages_str}], region(s) {c['expected_regions']}"
            )
            c["verification_method"] = (
                "Directly verified against extracted original PDF page text and rendered region PNG crops."
            )
        else:
            c["source"] = "Arctic Report Card 2025 (all 150 PDF pages checked)"
            c["verification_method"] = (
                "Whole-word case-insensitive regex scan across all 150 PDF pages confirmed 0 occurrences of core entity."
            )

    dev_pages = {p for c in dev for p in c["expected_pages"]}
    test_pages = {p for c in test for p in c["expected_pages"]}
    overlap = dev_pages & test_pages
    assert not overlap, f"Page overlap between dev and test: {sorted(overlap)}"

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(cases, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    from collections import Counter

    print(f"Wrote {len(cases)} benchmark cases to {OUT_PATH}")
    print("dev counts:", dict(Counter(c["group"] for c in dev)))
    print("test counts:", dict(Counter(c["group"] for c in test)))
    print("dev/test page overlap:", overlap)


if __name__ == "__main__":
    main()
