
# International vaccine data

## Files in this folder

- time_series_covid19_vaccine_global.csv: Contains time series data. Each row is uniquely defined by `Country_Region`, `UID`, and `Date`. Long format.
- time_series_covid19_vaccine_doses_admin_global.csv: Contains time series data. Each row is uniquely defined by `Country_Region`, `UID`, and `Date`. Wide format.
- data_dictionary.csv: Metric definitions
- readme.md: Description of contents and list of data sources

## Data sources
### Data updated after 9/21/2022
- On 9/21/2022, the CRC changed the data sources for all its vaccination data. The current data sources are:
  - US Centers for Disease Control and Prevention (CDC): https://covid.cdc.gov/covid-data-tracker/#vaccinations
  - Our World in Data (OWiD): https://ourworldindata.org/covid-vaccinations
  - World Health Organization (WHO): https://covid19.who.int/who-data/vaccination-data.csv

### Data updated on 9/20/2022 and before
- Aggregated data sources:
  - US Centers for Disease Control and Prevention (CDC): https://covid.cdc.gov/covid-data-tracker/#vaccinations
  - Our World in Data (OWiD): https://ourworldindata.org/covid-vaccinations
  - World Health Organization (WHO): https://covid19.who.int/who-data/vaccination-data.csv

- Non-US data sources at the country/region (Admin0) level: The international vaccine data includes Doses_admin, People_partially_vaccinated, People_fully_vaccinated. If the country does not report a variable, or the variable appears to be stale, we compare with Our World in Data and pick the most up-to-date data between the sources to produce composited data.
  - Argentina: Ministry of Health: http://datos.salud.gob.ar/dataset/vacunas-contra-covid-19-dosis-aplicadas-en-la-republica-argentina/archivo/b4684dd9-3cb7-45f7-9c0e-086550013e22
  - Australia: COVID Live: https://covidlive.com.au/vaccinations
  - Austria: Department of Health: https://info.gesundheitsministerium.gv.at/?re=opendata
  - Bahrain: Ministry of Health: https://healthalert.gov.bh/en/
  - Bangladesh: Directorate General of Health Services: http://103.247.238.92/webportal/pages/covid19-vaccination-update.php
  - Belgium: Institute of Health (Sciensano): https://covid-vaccinatie.be/en
  - Bolivia: Ministry of Health and Sports: https://www.minsalud.gob.bo/
  - Brazil: Ministry of Health: https://qsprod.saude.gov.br/extensions/DEMAS_C19Vacina/DEMAS_C19Vacina.html
  - Bulgaria: Unified Information Portal: https://coronavirus.bg/bg/statistika
  - Canada: COVID-19 Tracker: https://covid19tracker.ca/vaccinationtracker.html
  - Chile: Government of Chile: https://www.gob.cl/yomevacuno/
  - Colombia: Ministry of Health and Social Protection: https://www.minsalud.gov.co/portada-covid-19.html
  - Costa Rica: Costa Rica Social Security: https://www.ccss.sa.cr/web/coronavirus/vacunacion
  - Czechia: Ministry of Health of the Czech Republic: https://onemocneni-aktualne.mzcr.cz/vakcinace-cr
  - Denmark: Statum Serum Institute: https://experience.arcgis.com/experience/1c7ff08f6cef4e2784df7532d16312f1
  - Dominican Republic: Government of the Dominican Republic: https://vacunate.gob.do/
  - El Salvador: Government of El Salvador: https://covid19.gob.sv/
  - Equatorial Guinea: Ministry of Health and Social Welfare Equatorial Guinea: https://guineasalud.org/estadisticas/
  - Estonia: The Health Board: https://www.terviseamet.ee/et/koroonaviirus/avaandmed
  - Faroe Islands: The Government of the Faroe Islands: https://corona.fo/
  - Finland: Department of Health Welfare: https://thl.fi/fi/web/infektiotaudit-ja-rokotukset/ajankohtaista/ajankohtaista-koronaviruksesta-covid-19/tilannekatsaus-koronaviruksesta and https://www.thl.fi/episeuranta/rokotukset/koronarokotusten_edistyminen.html
  - France: Government of France: https://www.data.gouv.fr/fr/datasets/donnees-relatives-aux-personnes-vaccinees-contre-la-covid-19-1/
  - Germany: Federal Ministry of Health: https://impfdashboard.de/
  - Ghana: Ghana Health Service: https://ghanahealthservice.org/covid19/
  - Greece: https://data.gov.gr
  - Greenland: Government of Greenland: https://corona.nun.gl/emner/statistik/antal_vaccinerede
  - Guatemala: Ministry of Public Health and Social Assistance: https://tablerocovid.mspas.gob.gt/
  - Honduras: Government of the Republic of Honduras: http://www.salud.gob.hn/site/index.php/vacunascovid
  - Hong Kong: Government of Hong Kong Special Administration Region: https://www.covidvaccine.gov.hk/en/dashboard
  - India: Government of India: https://www.mygov.in/covid-19
  - Indonesia: Ministry of Health of the Republic of Indonesia: https://vaksin.kemkes.go.id/#/vaccines
  - Isle of Man: Isle of Man Government: https://covid19.gov.im/general-information/covid-19-vaccination-statistics/
  - Israel: Israel Ministry of Health: https://datadashboard.health.gov.il/COVID-19/general
  - Ireland: Government of Ireland: https://covid19ireland-geohive.hub.arcgis.com/ 
  - Italy: Ministry of Health: https://www.governo.it/it/cscovid19/report-vaccini/
  - Japan: Prime Minister of Japan and His Cabinet: https://www.kantei.go.jp/jp/headline/kansensho/vaccine.html 
  - Jordan: Ministry of Health: https://corona.moh.gov.jo/en
  - Kazakhstan: Ministry of Health: https://www.coronavirus2020.kz
  - Kenya: Ministry of Health: https://www.health.go.ke/#1621663315215-d6245403-4901
  - Latvia: National Health Service: https://data.gov.lv/dati/eng/dataset/covid19-vakcinacijas#
  - Lebanon: https://impact.cib.gov.lb/home/dashboard/vaccine
  - Lithuania: State Data Management Information System: https://experience.arcgis.com/experience/cab84dcfe0464c2a8050a78f817924ca/page/page_3/
  - Luxembourg: Ministry of Health: https://data.public.lu/fr/datasets/covid-19-rapports-journaliers/#_
  - Malaysia: Malaysia's National Covid-19 Immunization Program: https://github.com/CITF-Malaysia/citf-public/tree/main/vaccination
  - Malta: https://raw.githubusercontent.com/COVID19-Malta/
  - Mexico: Government of Mexico: https://coronavirus.gob.mx/noticias/
  - Mongolia: https://visual.ikon.mn/
  - Montenegro: Government of Montenegro: https://www.covidodgovor.me/
  - Nepal: Ministry of Health and Population: https://covid19.mohp.gov.np/situation-report
  - New Zealand: Ministry of Health: https://www.health.govt.nz/our-work/diseases-and-conditions/covid-19-novel-coronavirus/covid-19-data-and-statistics/covid-19-vaccine-data
  - Netherlands, The: Central Government of the Netherlands: https://coronadashboard.rijksoverheid.nl/landelijk/vaccinaties
  - Norway: Public Health Institute: https://www.fhi.no/sv/vaksine/koronavaksinasjonsprogrammet/koronavaksinasjonsstatistikk/#table-pagination-66720844
  - Pakistan: Government of Pakistan: https://ncoc.gov.pk/covid-vaccination-en.php
  - Paraguay: Ministry of Health and Social Wellness: https://www.vacunate.gov.py/index-listado-vacunados.html
  - Peru: Ministry of Health: https://gis.minsa.gob.pe/GisVisorVacunados/
  - Poland: Government of Poland: https://www.gov.pl/web/szczepimysie/raport-szczepien-przeciwko-covid-19
  - Portugal: General Health Management: https://esriportugal.maps.arcgis.com/apps/opsdashboard/index.html#/acf023da9a0b4f9dbb2332c13f635829
  - Qatar: Ministry of Public Health: https://covid19.moph.gov.qa/EN/Pages/Vaccination-Program-Data.aspx
  - Romania: Ministry of Health: https://datelazi.ro/
  - Russia: Gogov: https://gogov.ru/articles/covid-v-stats 
  - Saint Lucia: Ministry of Health: https://www.covid19response.lc
  - San Marino: Social Security Institute: https://vaccinocovid.iss.sm/
  - Saudi Arabia: Ministry of Health: https://covid19.moh.gov.sa/
  - Senegal: Ministry of Health and Social Action: https://cartosantesen.maps.arcgis.com/apps/dashboards/260c7842a77a48c191bf51c8b0a1d3f6
  - Singapore: Ministry of Health: https://www.moh.gov.sg/covid-19
  - Slovenia: https://covid-19.sledilnik.org/en/data
  - South Africa: Republic of South Africa Department of Health: https://sacoronavirus.co.za/
  - South Korea: Korea Disease Control and Prevention Agency: https://ncv.kdca.go.kr/mainStatus.es?mid=a11702000000
  - Spain: Ministry of Health: https://www.mscbs.gob.es/profesionales/saludPublica/ccayes/alertasActual/nCov/pbiVacunacion.htm
  - Sri Lanka: Ministry of Health Epidemiology Unit: http://www.epid.gov.lk/web/index.php?option=com_content&view=article&id=225&lang=en
  - Suriname: Ministry of Health: https://laatjevaccineren.sr/
  - Sweden: Public Health Authority: https://www.folkhalsomyndigheten.se/
  - Switzerland: Federal Office of Public Health FOPH: https://www.covid19.admin.ch/en/epidemiologic/vacc-doses
  - Thailand: Department of Disease Control: https://ddc.moph.go.th/vaccine-covid19/diaryReport
  - Turkey: Ministry of Health: https://covid19asi.saglik.gov.tr/
  - Uganda: Ministry of Health: https://www.health.go.ug/covid/
  - Ukraine: Government of Ukraine: https://health-security.rnbo.gov.ua/vaccination
  - United Arab Emirates: Supreme Council for National Security: https://covid19.ncema.gov.ae/en
  - United Kingdom: United Kingdom Government: https://coronavirus.data.gov.uk/
  - Uruguay: Ministry of Public Health: https://monitor.uruguaysevacuna.gub.uy/
  - Zambia: Zambia National Public health Institute: https://rtc-planning.maps.arcgis.com/apps/dashboards/3b3a01c1d8444932ba075fb44b119b63
