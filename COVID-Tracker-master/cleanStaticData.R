#*************************
# ---- Data cleaning ----
#************************
library(openxlsx)
library(dplyr)
library(tigris)
library(purrr)
library(stringr)

#Get all FIPS codes
colnames(fips_codes) = str_to_sentence(colnames(fips_codes))
fipsData = fips_codes %>% mutate(fips = paste0(State_code, County_code)) 


#Get population data per fips
popData = read.csv("data/est2019-allData.csv") %>% 
  #Make sure the fips is a character with leading 0 is needed
  mutate(fips = as.character(fips), fips = ifelse(nchar(fips) < 5, paste0(0, fips), fips))
  

fipsData = fipsData %>% left_join(popData)

#Load the region data and add the full FIPS
metroData = read.xlsx("data/metro_fips_codes.xlsx") %>% 
  mutate(fips = paste0(FIPS.State.Code, FIPS.County.Code)) %>% 
  select(-`County/County.Equivalent`, -FIPS.County.Code, -FIPS.State.Code, -State.Name)
  
  
fipsData = fipsData %>% left_join(metroData, by = "fips") %>% mutate(FIPS = fips) %>% select(-fips)

#Get the ZIP to FIPS data
#https://www.census.gov/geographies/reference-files/time-series/geo/relationship-files.html#par_textimage_674173622
zipToFIPS = read.table("data/zcta.txt", sep = ",", header = T, colClasses = "character") %>% 
  select(ZIP = ZCTA5, STATE, COUNTY) %>% mutate(FIPS = paste0(STATE, COUNTY)) %>% 
  distinct() %>% select(ZIP, FIPS)

# test = fipsData %>% filter(str_detect(County, "Kendall"), State.Name == "Illinois") %>% pull(FIPS)
# paste(test, collapse = ", ")
# fipsData %>% filter(FIPS %in% c("36005", "36047", "36061", "36081", "36085"))
# fipsData %>% filter(FIPS %in% c("29037", "29047", "29095", "29165"))
#c("17031", "17043", "17063", "17111", "17197", "17037", "17089", "17093", "18073", "18089", "18111", "18127", "17097", "55059")

# test = fipsData %>% filter(FIPS == "29095")
# paste(unlist(test), collapse = "', '")

#Remove the NYC counties we'll merge
NYCfips = c("36005", "36047", "36061", "36081", "36085")
fipsData = fipsData %>% filter(!FIPS %in% NYCfips)

#Manually add NYC
fipsData = rbind(fipsData, 
      list("NY", "36", "New York", "124", "Bronx-Kings-New York\nQueens-Richmont", 8336817, "35620",
           "35614", "4080", "New York-Newark-Jersey City, NY-NJ-PA", "Metropolitan Statistical Area",
           "New York-Jersey City-White Plains, NY-NJ", "New York-Newark, NY-NJ-CT-PA", "Central", "36124"))

#Update the ZIPs FIPS
zipToFIPS[zipToFIPS$FIPS %in% NYCfips, "FIPS"] = "36124"

#Manually add Kansas City
fipsData = rbind(fipsData, 
                 list('MO', '29', 'Missouri', '511', 'Kansas City', 703011, '28140', 
                      'NA', '312', 'Kansas City, MO-KS', 'Metropolitan Statistical Area', 
                      'NA', 'Kansas City-Overland Park-Kansas City, MO-KS', 'Central', '29511'))

#Add all the zip codes for the counties to the list with the new fips
zipToFIPS = rbind(
  zipToFIPS,
  data.frame(
    ZIP = zipToFIPS %>% filter(FIPS %in% c("29037", "29047", "29095", "29165")) %>% 
      pull(ZIP),
    FIPS = '29511'
  )
)


#Edit Cook county Chicago
fipsData[fipsData$FIPS == "17031", "County"] = "Cook (whole of Chicago)"
# fipsData[fipsData$FIPS == "17031", "County"] = "Cook-DuPage-Grundy-McHenry\nWill-DeKalb-Kane-Kendall\nJasper-Lake-Newton\nPorter-Lake-Kenosha"
fipsData[fipsData$FIPS == "17031", "POPESTIMATE2019"] = 9458539


#Insert unknown counties
unknownCounties = read.csv("data/unknownCounties.csv", stringsAsFactors = F, colClasses = "character")

unknownCounties = map_df(unknownCounties$stateName, function(myState){
  myState = unknownCounties %>% filter(stateName == myState)
  myState = data.frame(myState$state, str_extract(myState$FIPS, "^.."), myState$stateName, "000", 
                       "Unknown", 0, NA, NA, NA, NA, "Metropolitan Statistical Area", NA, NA, 
                       "Unknown", myState$FIPS)
  
  myState
})
colnames(unknownCounties) = colnames(fipsData)

#Add missing state data in the unknown counties
unknownCounties[unknownCounties$State %in% c("AS","GU", "MP", "PR", "UM", "VI"),"POPESTIMATE2019"] = 
  c(55689 , 167772, 51994, 2933408, 300, 104578)


fipsData = rbind(fipsData, unknownCounties)
fipsData = fipsData %>% mutate(Country = "USA") %>% filter(County_code != "515", !is.na(POPESTIMATE2019))

#Save
write.csv(fipsData, "data/fipsData.csv", row.names = F)
write.csv(zipToFIPS, "data/zipToFIPS.csv", row.names = F)
