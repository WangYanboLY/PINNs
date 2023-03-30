wd <- setwd("/Users/wangyanbo/PINNs/state_wise_data")
library(purrr)
library(readr)

file_names <- list.files(path = wd, pattern = ".csv", full.names = TRUE)
my_data_list <- map(file_names, read_csv)

my_data_list <- lapply(my_data_list, function(df) {
  df[, c("Confirmed", "Deaths", "Recovered", "Date")]
})

list2env(setNames(my_data_list, tools::file_path_sans_ext(basename(file_names))), envir = .GlobalEnv)
recovered_counts <- sapply(my_data_list, function(df) {
  sum(!is.na(df$Recovered))
})

