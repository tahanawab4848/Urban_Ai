# Dataset Citation & Provenance

## Primary Dataset
- **Title:** Air Quality Data in India (2015–2020)
- **Primary Source / Host:** Kaggle (`rohanrao/air-quality-data-in-india`)
- **Primary File:** `city_day.csv`
- **Curator:** Rohan Rao
- **License:** Creative Commons CC0: Public Domain
- **DOI / URL:** https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india
- **Data Origin Authority:** Central Pollution Control Board (CPCB), Ministry of Environment, Forest and Climate Change, Government of India (https://cpcb.nic.in)

## Attributes / Schema
The dataset provides daily ambient air quality records across major Indian metropolitan cities (e.g., Delhi, Bengaluru, Hyderabad, Kolkata, Mumbai, Ahmedabad, Chennai):
1. `City`: Name of the urban center
2. `Date`: Daily monitoring date (YYYY-MM-DD, covering 2015-01-01 through 2020-07-01)
3. `PM2.5`: Particulate matter $< 2.5\ \mu\text{m}$ ($\mu\text{g/m}^3$)
4. `PM10`: Particulate matter $< 10\ \mu\text{m}$ ($\mu\text{g/m}^3$)
5. `NO`: Nitric Oxide ($\mu\text{g/m}^3$)
6. `NO2`: Nitrogen Dioxide ($\mu\text{g/m}^3$)
7. `NOx`: Nitrogen Oxides total ($\mu\text{g/m}^3$)
8. `NH3`: Ammonia ($\mu\text{g/m}^3$)
9. `CO`: Carbon Monoxide ($\text{mg/m}^3$)
10. `SO2`: Sulfur Dioxide ($\mu\text{g/m}^3$)
11. `O3`: Ground-level Ozone ($\mu\text{g/m}^3$)
12. `Benzene`: Hydrocarbon volatile organic compound ($\mu\text{g/m}^3$)
13. `Toluene`: Hydrocarbon volatile organic compound ($\mu\text{g/m}^3$)
14. `Xylene`: Hydrocarbon volatile organic compound ($\mu\text{g/m}^3$)
15. `AQI`: Computed numeric Air Quality Index (0 to $500+$)
16. `AQI_Bucket`: Categorical classification according to CPCB/NAQI standards:
    - *Good* (0–50)
    - *Satisfactory* (51–100)
    - *Moderate* (101–200)
    - *Poor* (201–300)
    - *Very Poor* (301–400)
    - *Severe* (401–500+)

## Secondary / Benchmark Reference Standard
- **Central Pollution Control Board (CPCB) National Air Quality Index (NAQI):**
  - CPCB Report: *National Air Quality Index* (NAQI), Ministry of Environment, Forest & Climate Change, Govt. of India (2014/2015).
  - Breakpoints and Sub-index calculation algorithms specified by the technical steering committee.

## Formal BibTeX Citation
```bibtex
@misc{rao2020airqualityindia,
  author = {Rao, Rohan},
  title = {Air Quality Data in India (2015-2020)},
  year = {2020},
  publisher = {Kaggle},
  howpublished = {\url{https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india}},
  note = {Data sourced originally from Central Pollution Control Board (CPCB), India. CC0 Public Domain License}
}

@techreport{cpcb2014naqi,
  author = {{Central Pollution Control Board}},
  title = {National Air Quality Index (NAQI)},
  institution = {Ministry of Environment, Forests and Climate Change, Government of India},
  year = {2014},
  address = {New Delhi, India},
  url = {https://cpcb.nic.in/upload/NAQI_Report.pdf}
}
```
