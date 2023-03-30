library(dplyr)
library(tidyr)

death <- read.csv("/Users/wangyanbo/PINNs/COVID-19-master-2/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv")
useful_death_by_county <- death %>% 
  select(-c(UID, iso2, iso3, code3, FIPS, Lat, Long_, Country_Region,Combined_Key)) %>% 
  pivot_longer(cols = starts_with("X"), names_to = "day", values_to = "daily_deaths") 

View(death)

View(useful_death_by_county)


CA_death1 <- useful_death_by_county %>% 
  filter(Province_State == "California", day == "X1.1.21")

View(CA_death1)


CA_death <- useful_death_by_county %>% 
  filter(Province_State == "California") %>% 
  group_by(day) %>% 
  summarise(sum(daily_deaths))

View(CA_death)



data_in_the_paper <- read.csv("/Users/wangyanbo/PINNs/EulerRK_DNN2covid-main/data/minnesota.csv")
setwd("/Users/wangyanbo/PINNs/COVID-Tracker-master/data")

data_quality <- read.csv("dataReliability.csv")
hospitalData <- read.csv("hospitalData.csv")
covid_project_data <- read.csv("covidProjectData.csv")
NYTdata <- read.csv("NYTdata.csv")
zcta <- read.table("zcta.txt", header = TRUE, sep = ",")
est2019 <- read.csv("est2019-alldata.csv")
us_counties <- read.csv("us-counties.csv")


# The data reliability from CA is only B. The recovered patients
# are not recorded. 
# First I want to find the state with the most recorded recovered data.
# 
recovered_data <- hospitalData %>% 
  group_by(state) %>% 
  summarize(non_na_recovered = sum(!is.na(recovered)))
state_with_most_recovered <- recovered_data %>%
  arrange(desc(non_na_recovered)) %>%
  head(1) %>%
  pull(state)

Minnesota_daily <- us_counties %>%
  filter(state == "Minnesota") %>% 
  group_by(date) %>%
  summarize(daily_deaths = sum(deaths), daily_cases = sum(cases)) %>%
  ungroup()
 
Minnesota_counties <- us_counties %>% 
  filter(state == "Minnesota")
# Secondly I calculate the daily IRD data based on the cumulated data provided.
hospitalDataUseful <- select(hospitalData, c(date, state, positive, recovered, death))
AR_daily_data<- hospitalDataUseful %>% filter(state == "AR")



