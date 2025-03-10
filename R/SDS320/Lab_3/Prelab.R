data <- read.csv("TTestHandout_maze.csv")

data %>% 
  group_by(Group) %>% 
  filter(Group == "odor")
