#Pop Summary - Comp Bio Club
#I had some models running in Python so I used R :skull:

library(tidyverse)
library(ggplot2)

pop <- read.table("pop_data.txt", header = TRUE)
fam <- read.table("fam_data.txt", header = TRUE, fill = TRUE)

pairs <- pop %>% 
  mutate(count = 1) %>% 
  group_by(pop, super_pop) %>% 
  summarize(sum(count)) %>% 
  select(pop, super_pop)

joined <- full_join(fam, pairs, by = join_by(Population == pop),
          relationship = "many-to-one") %>% 
  mutate(count = 1)

#Compute the number and % of individuals per ancestry group & superpopulation 
#pop super_pop

pop_summary <- 
  joined %>%
  group_by(Population) %>% 
  summarize(num_indiv = sum(count), percent = sum(count)/nrow(joined)) %>% 
  full_join(pairs, by = join_by(Population == pop),
            relationship = "many-to-one")

super_summary <-
  joined %>%
  group_by(super_pop) %>% 
  summarize(super_num_indiv = sum(count), super_percent = sum(count)/nrow(joined))

population_summary <-
  full_join(pop_summary, super_summary, by = super_pop)