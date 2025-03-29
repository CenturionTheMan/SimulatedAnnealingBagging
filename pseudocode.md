**Input** \
data: data to create bags for\
bags_amount: amount of bags to create\
bags_size_ratio: ratio of the size of the bags to the size of the data\
t0: initial temperature\
alpha: temperature decay rate\
min_temp: minimum temperature\
max_iter: maximum number of iterations\

**Output**

1. bags <- create initial bags
2. cur_models <- train models on cur_bags (decision trees)
3. cur_fitness <- evaluate cur_models on data
4. best_models <- cur_models
5. best_fitness <- cur_fitness
6. t <- t0
7. iter <- 0
8. **while** t > min_temp and iter < max_iter:
   1. bags <- get new bags by perturbing cur_bags (solution in the neighborhood)
   2. new_models <- train models on new_bags
   3. new_fitness <- evaluate new_models on data
   4. **if** new_fitness > best_fitness:
      1. best_models <- new_models
      2. best_fitness <- new_fitness
   5. **if** new_fitness > cur_fitness:
      1. cur_models <- new_models
      2. cur_fitness <- new_fitness
   6. **else**:
      1. delta <- new_fitness - cur_fitness
      2. p <- exp(delta / t)
      3. **if** random() < p:
         1. cur_models <- new_models
         2. cur_fitness <- new_fitness
   7. t <- t \* alpha
   8. iter <- iter + 1
9. **return** best_models
