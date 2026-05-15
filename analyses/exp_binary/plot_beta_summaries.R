#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)

get_arg <- function(i, default) {
  if (length(args) >= i && nzchar(args[[i]])) args[[i]] else default
}

beta_arg <- get_arg(1, "0.01,0.1,1.0,2.0,4.0,6.0,8.0,10.0")
lambda_arg <- get_arg(2, "1.0")
alpha_arg <- get_arg(3, "0.0")
opportunity_arg <- get_arg(4, "0.0")
input_dir <- get_arg(5, "outputs/simulations")
results_dir <- get_arg(6, "results")
tree_size <- as.integer(get_arg(7, "2"))
input_type <- get_arg(8, "binary")
expansion_decision_version <- get_arg(9, "decoder")

normalize_expansion_decision_version <- function(version) {
  version_key <- tolower(trimws(as.character(version)))
  aliases <- c(
    "1" = "decoder",
    "decoder" = "decoder",
    "after_decoder" = "decoder",
    "2" = "lstm",
    "lstm" = "lstm",
    "after_lstm" = "lstm",
    "3" = "pre_lstm",
    "pre_lstm" = "pre_lstm",
    "before_lstm" = "pre_lstm"
  )
  if (!version_key %in% names(aliases)) {
    stop(sprintf(
      "expansion_decision_version must be one of: %s. Got %s.",
      paste(names(aliases), collapse = ", "),
      version
    ))
  }
  unname(aliases[[version_key]])
}

expansion_decision_version <- normalize_expansion_decision_version(expansion_decision_version)

dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)

beta_values <- trimws(strsplit(beta_arg, ",")[[1]])
opportunity_values <- trimws(strsplit(opportunity_arg, ",")[[1]])
seeds <- 1:5

arg_label <- function(values) {
  label <- paste(values, collapse = "_")
  gsub("[^A-Za-z0-9._-]+", "_", label)
}

format_plot_values <- function(values) {
  value_nums <- suppressWarnings(as.numeric(values))
  vapply(seq_along(values), function(i) {
    if (is.na(value_nums[[i]])) {
      return(as.character(values[[i]]))
    }
    format(signif(value_nums[[i]], 5), scientific = FALSE, trim = TRUE)
  }, character(1))
}

opportunity_label <- arg_label(opportunity_values)
beta_label <- arg_label(beta_values)
expansion_label <- arg_label(expansion_decision_version)

drop_unnamed_index_columns <- function(dat) {
  unnamed_cols <- names(dat) %in% c("", "...1", "X", "X1")
  if (any(unnamed_cols)) {
    dat <- dat[, !unnamed_cols, drop = FALSE]
  }
  dat
}

value_candidates <- function(x) {
  x_chr <- as.character(x)
  x_num <- suppressWarnings(as.numeric(x_chr))
  candidates <- x_chr
  if (!is.na(x_num)) {
    candidates <- c(
      candidates,
      sprintf("%.1f", x_num),
      sprintf("%.2f", x_num),
      format(x_num, scientific = FALSE, trim = TRUE)
    )
  }
  unique(candidates)
}

simulation_path <- function(lambda_value, alpha_value, beta_value, opportunity_value, seed) {
  lambda_candidates <- value_candidates(lambda_value)
  alpha_candidates <- value_candidates(alpha_value)
  beta_candidates <- value_candidates(beta_value)
  opportunity_candidates <- value_candidates(opportunity_value)

  for (lambda_candidate in lambda_candidates) {
    for (alpha_candidate in alpha_candidates) {
      for (beta_candidate in beta_candidates) {
        for (opportunity_candidate in opportunity_candidates) {
          file_names <- c(
            sprintf(
              "lambda_%s_alpha_%s_beta_%s_opportunity_%s_expansion_%s_seed_%d_%dn_%s.csv",
              lambda_candidate, alpha_candidate, beta_candidate, opportunity_candidate,
              expansion_decision_version, seed, tree_size, input_type
            )
          )
          for (file_name in file_names) {
            file_path <- file.path(input_dir, file_name)
            if (file.exists(file_path)) {
              return(file_path)
            }
          }
        }
      }
    }
  }

  NA_character_
}

read_seed_file <- function(beta_value, opportunity_value, seed) {
  file_path <- simulation_path(lambda_arg, alpha_arg, beta_value, opportunity_value, seed)
  if (is.na(file_path)) {
    warning(sprintf(
      "Missing simulation file for beta=%s opportunity=%s seed=%d",
      beta_value, opportunity_value, seed
    ))
    return(NULL)
  }

  dat <- read.csv(file_path, stringsAsFactors = FALSE)
  dat <- drop_unnamed_index_columns(dat)
  dat$beta <- beta_value
  dat$opportunity <- opportunity_value
  dat$seed <- seed
  dat$file_path <- file_path
  dat$source_file <- file_path
  dat
}

loaded_data <- list()
for (beta_value in beta_values) {
  for (opportunity_value in opportunity_values) {
    for (seed in seeds) {
      seed_data <- read_seed_file(beta_value, opportunity_value, seed)
      if (!is.null(seed_data)) {
        loaded_data[[length(loaded_data) + 1]] <- seed_data
      }
    }
  }
}

all_data <- if (length(loaded_data) > 0) {
  do.call(rbind, loaded_data)
} else {
  NULL
}

if (is.null(all_data) || nrow(all_data) == 0) {
  stop("No simulation CSVs were found. Check beta/lambda/alpha values and input_dir.")
}

if ("opportunity_cost" %in% names(all_data)) {
  requested_opportunity <- suppressWarnings(as.numeric(opportunity_values))
  if (all(!is.na(requested_opportunity))) {
    row_opportunity <- suppressWarnings(as.numeric(all_data$opportunity_cost))
    matches_requested <- vapply(row_opportunity, function(x) {
      any(abs(x - requested_opportunity) < 1e-8)
    }, logical(1))
    all_data <- all_data[matches_requested, , drop = FALSE]
  }

  if (nrow(all_data) == 0) {
    stop(sprintf("No rows matched opportunity_cost=%s.", opportunity_arg))
  }
}

if ("expansion_decision_version" %in% names(all_data)) {
  all_data <- all_data[
    as.character(all_data$expansion_decision_version) == expansion_decision_version,
    ,
    drop = FALSE
  ]
  if (nrow(all_data) == 0) {
    stop(sprintf("No rows matched expansion_decision_version=%s.", expansion_decision_version))
  }
}

as_logical_col <- function(x) {
  if (is.logical(x)) {
    return(x)
  }
  tolower(as.character(x)) %in% c("true", "t", "1", "yes", "y", "stop")
}

build_current_stop_data <- function(dat) {
  reward_cols <- grep("^expanded_reward_t[0-9]+$", names(dat), value = TRUE)
  reward_cols <- reward_cols[order(as.integer(sub("^expanded_reward_t", "", reward_cols)))]
  stop_cols <- grep("^stop_t[0-9]+$", names(dat), value = TRUE)
  stop_cols <- stop_cols[order(as.integer(sub("^stop_t", "", stop_cols)))]

  if (length(reward_cols) == 0 || length(stop_cols) == 0) {
    stop(
      paste(
        "Cannot compute timestep-specific stop probabilities.",
        "Expected expanded_reward_t* and stop_t* columns. The loaded files only contain:",
        paste(names(dat), collapse = ", ")
      )
    )
  }

  n_steps <- min(length(reward_cols), length(stop_cols))
  if (n_steps < 2) {
    stop("Need at least two timesteps to compute P(stop at t+1 | reward observed at t).")
  }

  trial_id_cols <- intersect(c("beta", "opportunity", "seed", "graph"), names(dat))
  trial_cols <- unique(c(trial_id_cols, reward_cols[seq_len(n_steps)], stop_cols[seq_len(n_steps)]))
  trial_data <- unique(dat[, trial_cols, drop = FALSE])

  rows <- list()
  row_i <- 1
  for (i in seq_len(nrow(trial_data))) {
    for (t in seq_len(n_steps - 1)) {
      reward_t <- suppressWarnings(as.numeric(trial_data[[reward_cols[[t]]]][[i]]))
      if (!is.na(reward_t)) {
        rows[[row_i]] <- data.frame(
          beta = trial_data$beta[[i]],
          opportunity = trial_data$opportunity[[i]],
          seed = trial_data$seed[[i]],
          graph = trial_data$graph[[i]],
          reward_timestep = t,
          decision_timestep = t + 1,
          reward = reward_t,
          stop_current = as_logical_col(trial_data[[stop_cols[[t + 1]]]][[i]]),
          stringsAsFactors = FALSE
        )
        row_i <- row_i + 1
      }
    }
  }

  if (length(rows) == 0) {
    warning("No observed rewards were found before a current stop decision; stop-by-reward panels will be empty.")
    return(data.frame(
      beta = character(),
      opportunity = character(),
      seed = integer(),
      graph = integer(),
      reward_timestep = integer(),
      decision_timestep = integer(),
      reward = numeric(),
      stop_current = logical(),
      stringsAsFactors = FALSE
    ))
  }

  do.call(rbind, rows)
}

stop_data <- build_current_stop_data(all_data)

if (nrow(stop_data) > 0) {
  stop_summary <- aggregate(
    stop_current ~ beta + opportunity + reward_timestep + decision_timestep + reward,
    data = stop_data,
    FUN = mean
  )
  names(stop_summary)[names(stop_summary) == "stop_current"] <- "p_stop_current"

  stop_counts <- aggregate(
    stop_current ~ beta + opportunity + reward_timestep + decision_timestep + reward,
    data = stop_data,
    FUN = length
  )
  names(stop_counts)[names(stop_counts) == "stop_current"] <- "n"
  stop_summary <- merge(
    stop_summary,
    stop_counts,
    by = c("beta", "opportunity", "reward_timestep", "decision_timestep", "reward")
  )
  stop_summary$p_continue_current <- 1 - stop_summary$p_stop_current
} else {
  stop_summary <- data.frame(
    beta = character(),
    opportunity = character(),
    reward_timestep = integer(),
    decision_timestep = integer(),
    reward = numeric(),
    p_stop_current = numeric(),
    p_continue_current = numeric(),
    n = integer(),
    stringsAsFactors = FALSE
  )
}

if ("MI" %in% names(all_data)) {
  all_data$MI_value <- all_data$MI
} else if ("MI_cost" %in% names(all_data)) {
  all_data$MI_value <- all_data$MI_cost
} else {
  stop("Expected an MI or MI_cost column for the average V vs MI plot.")
}

v_mi_seed <- aggregate(
  cbind(V, MI_value) ~ beta + opportunity + seed,
  data = all_data,
  FUN = mean
)
v_mi_summary <- aggregate(
  cbind(V, MI_value) ~ beta + opportunity,
  data = v_mi_seed,
  FUN = mean
)

build_kl_summary <- function(dat) {
  kl_cols <- grep("^kl_d_t[0-9]+$", names(dat), value = TRUE)
  kl_cols <- kl_cols[order(as.integer(sub("^kl_d_t", "", kl_cols)))]

  if (length(kl_cols) == 0) {
    stop(
      paste(
        "Expected kl_d_t* columns for KL plotting, but none were found.",
        "Please re-run simulate.py with the updated model outputs."
      )
    )
  }

  trial_id_cols <- intersect(c("beta", "opportunity", "seed", "graph"), names(dat))
  trial_cols <- unique(c(trial_id_cols, kl_cols))
  trial_data <- unique(dat[, trial_cols, drop = FALSE])

  rows <- list()
  row_i <- 1
  for (i in seq_len(nrow(trial_data))) {
    for (kl_col in kl_cols) {
      timestep <- as.integer(sub("^kl_d_t", "", kl_col))
      kl_value <- suppressWarnings(as.numeric(trial_data[[kl_col]][[i]]))
      if (!is.na(kl_value)) {
        rows[[row_i]] <- data.frame(
          beta = trial_data$beta[[i]],
          opportunity = trial_data$opportunity[[i]],
          seed = trial_data$seed[[i]],
          graph = trial_data$graph[[i]],
          timestep = timestep,
          kl_d = kl_value,
          stringsAsFactors = FALSE
        )
        row_i <- row_i + 1
      }
    }
  }

  if (length(rows) == 0) {
    stop("KL columns were found, but all kl_d_t* values are NA.")
  }

  kl_data <- do.call(rbind, rows)
  aggregate(
    kl_d ~ beta + opportunity + timestep,
    data = kl_data,
    FUN = mean
  )
}

kl_summary <- build_kl_summary(all_data)

build_kl_by_reward_summary <- function(dat, timestep = 1) {
  reward_col <- sprintf("expanded_reward_t%d", timestep)
  kl_col <- sprintf("kl_d_t%d", timestep)

  missing_cols <- setdiff(c(reward_col, kl_col), names(dat))
  if (length(missing_cols) > 0) {
    stop(
      paste(
        "Cannot compute KL-by-reward summary. Missing columns:",
        paste(missing_cols, collapse = ", ")
      )
    )
  }

  trial_id_cols <- intersect(c("beta", "opportunity", "seed", "graph"), names(dat))
  trial_cols <- unique(c(trial_id_cols, reward_col, kl_col))
  trial_data <- unique(dat[, trial_cols, drop = FALSE])

  trial_data$reward <- suppressWarnings(as.numeric(trial_data[[reward_col]]))
  trial_data$kl_d <- suppressWarnings(as.numeric(trial_data[[kl_col]]))
  trial_data <- trial_data[
    !is.na(trial_data$reward) & !is.na(trial_data$kl_d),
    ,
    drop = FALSE
  ]

  if (nrow(trial_data) == 0) {
    warning(sprintf("No non-NA %s/%s pairs were found; KL-by-reward panel will be empty.", reward_col, kl_col))
    return(data.frame(
      beta = character(),
      opportunity = character(),
      reward = numeric(),
      kl_d = numeric(),
      n = integer(),
      stringsAsFactors = FALSE
    ))
  }

  kl_reward_summary <- aggregate(
    kl_d ~ beta + opportunity + reward,
    data = trial_data,
    FUN = mean
  )

  kl_reward_counts <- aggregate(
    kl_d ~ beta + opportunity + reward,
    data = trial_data,
    FUN = length
  )
  names(kl_reward_counts)[names(kl_reward_counts) == "kl_d"] <- "n"

  merge(
    kl_reward_summary,
    kl_reward_counts,
    by = c("beta", "opportunity", "reward")
  )
}

kl_reward_t1_summary <- build_kl_by_reward_summary(all_data, timestep = 1)

build_kl_by_reward_all_summary <- function(dat) {
  reward_cols <- grep("^expanded_reward_t[0-9]+$", names(dat), value = TRUE)
  reward_cols <- reward_cols[order(as.integer(sub("^expanded_reward_t", "", reward_cols)))]
  kl_cols <- grep("^kl_d_t[0-9]+$", names(dat), value = TRUE)
  kl_cols <- kl_cols[order(as.integer(sub("^kl_d_t", "", kl_cols)))]
  n_steps <- min(length(reward_cols), length(kl_cols))

  if (n_steps == 0) {
    warning("No expanded_reward_t*/kl_d_t* pairs were found; KL-by-reward panels will be empty.")
    return(data.frame(
      beta = character(),
      opportunity = character(),
      timestep = integer(),
      reward = numeric(),
      kl_d = numeric(),
      n = integer(),
      stringsAsFactors = FALSE
    ))
  }

  trial_id_cols <- intersect(c("beta", "opportunity", "seed", "graph"), names(dat))
  trial_cols <- unique(c(trial_id_cols, reward_cols[seq_len(n_steps)], kl_cols[seq_len(n_steps)]))
  trial_data <- unique(dat[, trial_cols, drop = FALSE])

  rows <- list()
  row_i <- 1
  for (i in seq_len(nrow(trial_data))) {
    for (t in seq_len(n_steps)) {
      reward_value <- suppressWarnings(as.numeric(trial_data[[reward_cols[[t]]]][[i]]))
      kl_value <- suppressWarnings(as.numeric(trial_data[[kl_cols[[t]]]][[i]]))
      if (!is.na(reward_value) && !is.na(kl_value)) {
        rows[[row_i]] <- data.frame(
          beta = trial_data$beta[[i]],
          opportunity = trial_data$opportunity[[i]],
          timestep = t,
          reward = reward_value,
          kl_d = kl_value,
          stringsAsFactors = FALSE
        )
        row_i <- row_i + 1
      }
    }
  }

  if (length(rows) == 0) {
    warning("No non-NA expanded_reward_t*/kl_d_t* pairs were found; KL-by-reward panels will be empty.")
    return(data.frame(
      beta = character(),
      opportunity = character(),
      timestep = integer(),
      reward = numeric(),
      kl_d = numeric(),
      n = integer(),
      stringsAsFactors = FALSE
    ))
  }

  kl_reward_data <- do.call(rbind, rows)
  kl_reward_summary <- aggregate(
    kl_d ~ beta + opportunity + timestep + reward,
    data = kl_reward_data,
    FUN = mean
  )
  kl_reward_counts <- aggregate(
    kl_d ~ beta + opportunity + timestep + reward,
    data = kl_reward_data,
    FUN = length
  )
  names(kl_reward_counts)[names(kl_reward_counts) == "kl_d"] <- "n"
  merge(
    kl_reward_summary,
    kl_reward_counts,
    by = c("beta", "opportunity", "timestep", "reward")
  )
}

round_to_reward <- function(x, reward_values) {
  vapply(x, function(value) {
    if (is.na(value) || length(reward_values) == 0) {
      return(NA_real_)
    }
    reward_values[[which.min(abs(reward_values - value))]]
  }, numeric(1))
}

build_reconstruction_accuracy_summary <- function(dat) {
  estimate_cols <- grep("^estimated_reward_t[0-9]+$", names(dat), value = TRUE)
  estimate_cols <- estimate_cols[order(as.integer(sub("^estimated_reward_t", "", estimate_cols)))]

  if (length(estimate_cols) == 0 || !"actual_reward" %in% names(dat)) {
    warning("No estimated_reward_t* and actual_reward columns were found; reconstruction accuracy panels will be empty.")
    return(data.frame(
      beta = character(),
      opportunity = character(),
      timestep = integer(),
      reward = numeric(),
      accuracy = numeric(),
      n = integer(),
      stringsAsFactors = FALSE
    ))
  }

  reward_values <- sort(unique(suppressWarnings(as.numeric(dat$actual_reward))))
  reward_values <- reward_values[!is.na(reward_values)]
  id_cols <- intersect(c("beta", "opportunity", "seed", "graph", "node", "actual_reward"), names(dat))
  node_data <- unique(dat[, unique(c(id_cols, estimate_cols)), drop = FALSE])

  rows <- list()
  row_i <- 1
  for (i in seq_len(nrow(node_data))) {
    actual_reward <- suppressWarnings(as.numeric(node_data$actual_reward[[i]]))
    if (is.na(actual_reward)) {
      next
    }
    for (estimate_col in estimate_cols) {
      timestep <- as.integer(sub("^estimated_reward_t", "", estimate_col))
      estimate <- suppressWarnings(as.numeric(node_data[[estimate_col]][[i]]))
      if (!is.na(estimate)) {
        predicted_reward <- round_to_reward(estimate, reward_values)
        rows[[row_i]] <- data.frame(
          beta = node_data$beta[[i]],
          opportunity = node_data$opportunity[[i]],
          timestep = timestep,
          reward = actual_reward,
          accurate = as.numeric(!is.na(predicted_reward) && predicted_reward == actual_reward),
          stringsAsFactors = FALSE
        )
        row_i <- row_i + 1
      }
    }
  }

  if (length(rows) == 0) {
    warning("No non-NA estimated rewards were found; reconstruction accuracy panels will be empty.")
    return(data.frame(
      beta = character(),
      opportunity = character(),
      timestep = integer(),
      reward = numeric(),
      accuracy = numeric(),
      n = integer(),
      stringsAsFactors = FALSE
    ))
  }

  recon_data <- do.call(rbind, rows)
  recon_summary <- aggregate(
    accurate ~ beta + opportunity + timestep + reward,
    data = recon_data,
    FUN = mean
  )
  names(recon_summary)[names(recon_summary) == "accurate"] <- "accuracy"
  recon_counts <- aggregate(
    accurate ~ beta + opportunity + timestep + reward,
    data = recon_data,
    FUN = length
  )
  names(recon_counts)[names(recon_counts) == "accurate"] <- "n"
  merge(
    recon_summary,
    recon_counts,
    by = c("beta", "opportunity", "timestep", "reward")
  )
}

get_trial_stop_data <- function(dat) {
  stop_cols <- grep("^stop_t[0-9]+$", names(dat), value = TRUE)
  stop_cols <- stop_cols[order(as.integer(sub("^stop_t", "", stop_cols)))]
  trial_id_cols <- intersect(c("beta", "opportunity", "seed", "graph", "chosen_path"), names(dat))
  trial_data <- unique(dat[, unique(c(trial_id_cols, stop_cols)), drop = FALSE])

  first_stop <- rep(NA_integer_, nrow(trial_data))
  for (i in seq_len(nrow(trial_data))) {
    stop_vec <- as_logical_col(unlist(trial_data[i, stop_cols, drop = TRUE]))
    stop_at <- which(stop_vec)
    if (length(stop_at) > 0) {
      first_stop[[i]] <- stop_at[[1]]
    }
  }
  trial_data$stop_timestep <- first_stop
  trial_data
}

chosen_node_from_path <- function(chosen_path) {
  chosen_path <- suppressWarnings(as.numeric(chosen_path))
  if (tree_size == 2) {
    return(chosen_path + 1)
  }
  chosen_path
}

build_choice_by_stop_summary <- function(dat) {
  required_cols <- c("chosen_path", "node", "actual_reward")
  missing_cols <- setdiff(required_cols, names(dat))
  if (length(missing_cols) > 0) {
    warning(sprintf("Cannot build choice-by-stop summary. Missing columns: %s", paste(missing_cols, collapse = ", ")))
    return(data.frame(
      beta = character(),
      opportunity = character(),
      stop_timestep = integer(),
      reward = numeric(),
      p_choose = numeric(),
      n = integer(),
      stringsAsFactors = FALSE
    ))
  }

  trial_stop <- get_trial_stop_data(dat)
  node_cols <- intersect(c("beta", "opportunity", "seed", "graph", "chosen_path", "node", "actual_reward"), names(dat))
  node_data <- unique(dat[, node_cols, drop = FALSE])
  node_data$node <- suppressWarnings(as.numeric(node_data$node))
  node_data$chosen_path <- suppressWarnings(as.numeric(node_data$chosen_path))
  node_data$chosen_node <- chosen_node_from_path(node_data$chosen_path)
  node_data$actual_reward <- suppressWarnings(as.numeric(node_data$actual_reward))
  chosen_rows <- node_data[
    !is.na(node_data$node) &
      !is.na(node_data$chosen_node) &
      node_data$node == node_data$chosen_node,
    ,
    drop = FALSE
  ]

  choice_data <- merge(
    trial_stop[, intersect(c("beta", "opportunity", "seed", "graph", "stop_timestep"), names(trial_stop)), drop = FALSE],
    chosen_rows[, intersect(c("beta", "opportunity", "seed", "graph", "actual_reward"), names(chosen_rows)), drop = FALSE],
    by = intersect(c("beta", "opportunity", "seed", "graph"), names(chosen_rows))
  )
  choice_data <- choice_data[!is.na(choice_data$stop_timestep) & !is.na(choice_data$actual_reward), , drop = FALSE]

  if (nrow(choice_data) == 0) {
    warning("No chosen node rewards were found for stopped trials; choice-by-stop panels will be empty.")
    return(data.frame(
      beta = character(),
      opportunity = character(),
      stop_timestep = integer(),
      reward = numeric(),
      p_choose = numeric(),
      n = integer(),
      stringsAsFactors = FALSE
    ))
  }

  choice_counts <- aggregate(
    graph ~ beta + opportunity + stop_timestep + actual_reward,
    data = choice_data,
    FUN = length
  )
  names(choice_counts) <- c("beta", "opportunity", "stop_timestep", "reward", "n")
  totals <- aggregate(
    n ~ beta + opportunity + stop_timestep,
    data = choice_counts,
    FUN = sum
  )
  names(totals)[names(totals) == "n"] <- "total_n"
  choice_summary <- merge(choice_counts, totals, by = c("beta", "opportunity", "stop_timestep"))
  choice_summary$p_choose <- choice_summary$n / choice_summary$total_n
  choice_summary[, c("beta", "opportunity", "stop_timestep", "reward", "p_choose", "n")]
}

build_choice_vs_other_summary <- function(dat, stop_timestep = 2) {
  required_cols <- c("chosen_path", "node", "actual_reward")
  missing_cols <- setdiff(required_cols, names(dat))
  if (length(missing_cols) > 0) {
    warning(sprintf("Cannot build choice-vs-other summary. Missing columns: %s", paste(missing_cols, collapse = ", ")))
    return(data.frame(
      beta = character(),
      opportunity = character(),
      other_reward = numeric(),
      reward = numeric(),
      p_choose = numeric(),
      n = integer(),
      stringsAsFactors = FALSE
    ))
  }

  trial_stop <- get_trial_stop_data(dat)
  trial_stop <- trial_stop[trial_stop$stop_timestep == stop_timestep, , drop = FALSE]
  if (nrow(trial_stop) == 0) {
    warning(sprintf("No trials stopped at timestep %d; choice-vs-other panels will be empty.", stop_timestep))
    return(data.frame(
      beta = character(),
      opportunity = character(),
      other_reward = numeric(),
      reward = numeric(),
      p_choose = numeric(),
      n = integer(),
      stringsAsFactors = FALSE
    ))
  }

  node_cols <- intersect(c("beta", "opportunity", "seed", "graph", "chosen_path", "node", "actual_reward"), names(dat))
  node_data <- unique(dat[, node_cols, drop = FALSE])
  node_data <- merge(
    node_data,
    trial_stop[, intersect(c("beta", "opportunity", "seed", "graph"), names(trial_stop)), drop = FALSE],
    by = intersect(c("beta", "opportunity", "seed", "graph"), names(node_data))
  )
  node_data$node <- suppressWarnings(as.numeric(node_data$node))
  node_data$chosen_path <- suppressWarnings(as.numeric(node_data$chosen_path))
  node_data$chosen_node <- chosen_node_from_path(node_data$chosen_path)
  node_data$actual_reward <- suppressWarnings(as.numeric(node_data$actual_reward))

  key_cols <- intersect(c("beta", "opportunity", "seed", "graph"), names(node_data))
  other_node_data <- node_data[, c(key_cols, "node", "actual_reward"), drop = FALSE]
  names(other_node_data)[names(other_node_data) == "node"] <- "other_node"
  names(other_node_data)[names(other_node_data) == "actual_reward"] <- "other_reward"
  choice_other_data <- merge(node_data, other_node_data, by = key_cols)
  choice_other_data <- choice_other_data[
    !is.na(choice_other_data$node) &
      !is.na(choice_other_data$other_node) &
      choice_other_data$node != choice_other_data$other_node &
      !is.na(choice_other_data$actual_reward) &
      !is.na(choice_other_data$other_reward),
    ,
    drop = FALSE
  ]
  choice_other_data$chose <- as.numeric(choice_other_data$node == choice_other_data$chosen_node)
  names(choice_other_data)[names(choice_other_data) == "actual_reward"] <- "reward"

  if (nrow(choice_other_data) == 0) {
    warning("No two-node stopped-at-timestep-2 trials were found; choice-vs-other panels will be empty.")
    return(data.frame(
      beta = character(),
      opportunity = character(),
      other_reward = numeric(),
      reward = numeric(),
      p_choose = numeric(),
      n = integer(),
      stringsAsFactors = FALSE
    ))
  }

  choice_other_summary <- aggregate(
    chose ~ beta + opportunity + other_reward + reward,
    data = choice_other_data,
    FUN = mean
  )
  names(choice_other_summary)[names(choice_other_summary) == "chose"] <- "p_choose"
  choice_other_counts <- aggregate(
    chose ~ beta + opportunity + other_reward + reward,
    data = choice_other_data,
    FUN = length
  )
  names(choice_other_counts)[names(choice_other_counts) == "chose"] <- "n"
  merge(
    choice_other_summary,
    choice_other_counts,
    by = c("beta", "opportunity", "other_reward", "reward")
  )
}

kl_reward_summary <- build_kl_by_reward_all_summary(all_data)
reconstruction_accuracy_summary <- build_reconstruction_accuracy_summary(all_data)
choice_stop_summary <- build_choice_by_stop_summary(all_data)
choice_other_summary <- build_choice_vs_other_summary(all_data, stop_timestep = 2)

beta_levels <- beta_values[beta_values %in% unique(all_data$beta)]
if (length(beta_levels) == 0) {
  beta_levels <- unique(all_data$beta)
}
opportunity_levels <- opportunity_values[opportunity_values %in% unique(all_data$opportunity)]
if (length(opportunity_levels) == 0) {
  opportunity_levels <- unique(all_data$opportunity)
}

color_by <- if (length(opportunity_levels) > 1 && length(beta_levels) == 1) {
  "opportunity"
} else {
  "beta"
}

if (length(beta_levels) > 1 && length(opportunity_levels) > 1) {
  warning("Both beta and opportunity have multiple values; using beta for color and opportunity for point/line style.")
}

color_levels <- if (identical(color_by, "opportunity")) opportunity_levels else beta_levels
color_level_labels <- if (identical(color_by, "opportunity")) {
  format_plot_values(color_levels)
} else {
  format_plot_values(color_levels)
}
palette_cols <- grDevices::hcl.colors(max(3, length(color_levels) + 2), palette = "Blues")
color_cols <- setNames(palette_cols[seq_along(color_levels)], color_levels)
beta_cols <- if (identical(color_by, "beta")) {
  color_cols
} else {
  setNames(rep("black", length(beta_levels)), beta_levels)
}
opportunity_cols <- if (identical(color_by, "opportunity")) {
  color_cols
} else {
  setNames(rep("black", length(opportunity_levels)), opportunity_levels)
}

series_color <- function(beta_value, opportunity_value) {
  if (identical(color_by, "opportunity")) {
    opportunity_cols[[as.character(opportunity_value)]]
  } else {
    beta_cols[[as.character(beta_value)]]
  }
}

color_legend_title <- if (identical(color_by, "opportunity")) "opportunity" else "beta"
color_legend_labels <- paste(color_legend_title, color_level_labels)
opportunity_pch_values <- c(19, 17, 15, 18, 8, 4, 3, 7, 9, 10)
opportunity_lty_values <- c(1, 2, 3, 4, 5, 6)
opportunity_pch <- setNames(
  rep(opportunity_pch_values, length.out = length(opportunity_levels)),
  opportunity_levels
)
opportunity_lty <- setNames(
  rep(opportunity_lty_values, length.out = length(opportunity_levels)),
  opportunity_levels
)

expand_range <- function(x, pad = 0.5) {
  x_range <- range(x, finite = TRUE)
  if (!all(is.finite(x_range))) {
    return(c(0, 1))
  }
  if (identical(x_range[[1]], x_range[[2]])) {
    return(c(x_range[[1]] - pad, x_range[[2]] + pad))
  }
  x_range
}

plot_parameter_legend <- function(position = "topright", include_style_legend = TRUE) {
  old_xpd <- par("xpd")
  par(xpd = NA)
  legend(
    position,
    inset = c(-0.32, 0),
    legend = color_legend_labels,
    col = color_cols[color_levels],
    pch = 19,
    lwd = 2,
    bty = "n"
  )
  if (include_style_legend && length(opportunity_levels) > 1 && !identical(color_by, "opportunity")) {
    legend(
      "bottomright",
      inset = c(-0.32, 0),
      legend = paste("opportunity", opportunity_levels),
      pch = opportunity_pch[opportunity_levels],
      lty = opportunity_lty[opportunity_levels],
      col = "black",
      bty = "n"
    )
  }
  par(xpd = old_xpd)
}

plot_reward_timestep_summary <- function(
  summary_data,
  value_col,
  ylab,
  main_prefix,
  empty_message,
  y_limits = NULL
) {
  timesteps <- sort(unique(summary_data$timestep))
  timesteps <- timesteps[!is.na(timesteps)]
  if (length(timesteps) == 0) {
    timesteps <- 1
  }

  old_par <- par(mfrow = c(1, length(timesteps)), mar = c(4.5, 4.5, 1, 8))
  for (timestep in timesteps) {
    panel_data <- summary_data[summary_data$timestep == timestep, , drop = FALSE]
    if (nrow(panel_data) == 0) {
      plot(
        NA,
        xlim = c(-0.1, 1.1),
        ylim = if (is.null(y_limits)) c(0, 1) else y_limits,
        xlab = "Reward",
        ylab = ylab,
        main = "",
        xaxt = "n"
      )
      grid()
      text(0.5, mean(par("usr")[3:4]), empty_message, cex = 0.9)
      next
    }

    y_values <- panel_data[[value_col]]
    plot(
      NA,
      xlim = expand_range(panel_data$reward, pad = 0.1),
      ylim = if (is.null(y_limits)) expand_range(y_values, pad = 0.05) else y_limits,
      xlab = "Reward",
      ylab = ylab,
      main = "",
      xaxt = "n"
    )
    axis(1, at = sort(unique(panel_data$reward)))
    grid()

    for (opportunity_value in opportunity_levels) {
      for (beta_value in beta_levels) {
        series_data <- panel_data[
          panel_data$beta == beta_value &
            panel_data$opportunity == opportunity_value,
          ,
          drop = FALSE
        ]
        series_data <- series_data[order(series_data$reward), , drop = FALSE]
        if (nrow(series_data) > 0) {
          lines(
            series_data$reward,
            series_data[[value_col]],
            type = "b",
            pch = opportunity_pch[[opportunity_value]],
            lwd = 2,
            lty = opportunity_lty[[opportunity_value]],
            col = series_color(beta_value, opportunity_value)
          )
        }
      }
    }
  }
  plot_parameter_legend()
  par(old_par)
}

continue_pdf <- file.path(
  results_dir,
  sprintf(
    "continue_probability_%s_lambda_%s_alpha_%s_beta_%s_opportunity_%s_expansion_%s_%dn.pdf",
    input_type, lambda_arg, alpha_arg, beta_label, opportunity_label, expansion_label, tree_size
  )
)

plot_stop_panel <- function(reward_timestep, opportunity_value = NULL) {
  panel_data <- stop_summary[
    stop_summary$reward_timestep == reward_timestep,
    ,
    drop = FALSE
  ]
  if (!is.null(opportunity_value)) {
    panel_data <- panel_data[
      panel_data$opportunity == opportunity_value,
      ,
      drop = FALSE
    ]
  }

  decision_timestep <- reward_timestep + 1
  opportunity_title <- if (is.null(opportunity_value)) {
    "all opportunities"
  } else {
    sprintf("opportunity %s", opportunity_value)
  }
  panel_title <- if (reward_timestep == 1) {
    sprintf("%s\nContinue at timestep 2 after reward at timestep 1", opportunity_title)
  } else if (reward_timestep == 2) {
    sprintf("%s\nContinue at timestep 3 after reward at timestep 2", opportunity_title)
  } else {
    sprintf(
      "%s\nContinue at timestep %d after reward at timestep %d",
      opportunity_title,
      decision_timestep,
      reward_timestep
    )
  }

  if (nrow(panel_data) == 0) {
  plot(
    NA,
    xlim = c(-0.1, 1.1),
    ylim = c(0, 1),
    xlab = "Observed reward",
    ylab = "P(continue at current timestep)",
      main = "",
      xaxt = "n"
    )
    axis(1, at = c(0, 1))
    grid()
    text(
      0.5,
      0.5,
      sprintf(
        "No stop_t%d decision exists\nafter reward_t%d",
        decision_timestep,
        reward_timestep
      ),
      cex = 0.9
    )
    return(invisible(NULL))
  }

  plot(
    NA,
    xlim = expand_range(panel_data$reward, pad = 0.1),
    ylim = c(0, 1),
    xlab = "Observed reward",
    ylab = "P(continue at current timestep)",
    main = "",
    xaxt = "n"
  )
  axis(1, at = sort(unique(panel_data$reward)))
  grid()

  for (opportunity_value_i in opportunity_levels) {
    for (beta_value in beta_levels) {
      beta_dat <- panel_data[
        panel_data$beta == beta_value &
          panel_data$opportunity == opportunity_value_i,
        ,
        drop = FALSE
      ]
      beta_dat <- beta_dat[order(beta_dat$reward), , drop = FALSE]
      if (nrow(beta_dat) > 0) {
        lines(
          beta_dat$reward,
          beta_dat$p_continue_current,
          type = "b",
          pch = opportunity_pch[[opportunity_value_i]],
          lwd = 2,
          lty = opportunity_lty[[opportunity_value_i]],
          col = series_color(beta_value, opportunity_value_i)
        )
      }
    }
  }
}

if (identical(color_by, "opportunity")) {
  pdf(continue_pdf, width = 14, height = 5)
  old_par <- par(mfrow = c(1, 2), mar = c(4.5, 4.5, 1, 8))
  plot_stop_panel(1)
  plot_stop_panel(2)
} else {
  pdf(continue_pdf, width = 14, height = max(5, 4.5 * length(opportunity_levels)))
  old_par <- par(mfrow = c(length(opportunity_levels), 2), mar = c(4.5, 4.5, 1, 8))
  for (opportunity_value in opportunity_levels) {
    plot_stop_panel(1, opportunity_value)
    plot_stop_panel(2, opportunity_value)
  }
}
plot_parameter_legend()
par(old_par)
dev.off()

v_mi_pdf <- file.path(
  results_dir,
  sprintf(
    "average_V_vs_MI_%s_lambda_%s_alpha_%s_beta_%s_opportunity_%s_expansion_%s_%dn.pdf",
    input_type, lambda_arg, alpha_arg, beta_label, opportunity_label, expansion_label, tree_size
  )
)

pdf(v_mi_pdf, width = 9, height = 5.5)
old_par <- par(mar = c(4.5, 4.5, 1, 8))
plot(
  v_mi_summary$MI_value,
  v_mi_summary$V,
  xlim = expand_range(v_mi_summary$MI_value, pad = 0.05),
  ylim = expand_range(v_mi_summary$V, pad = 0.05),
  xlab = "Average MI",
  ylab = "Average V",
  main = "",
  cex = 1.3,
  col = mapply(series_color, v_mi_summary$beta, v_mi_summary$opportunity),
  pch = opportunity_pch[as.character(v_mi_summary$opportunity)]
)
grid()
plot_parameter_legend()
par(old_par)
dev.off()

kl_pdf <- file.path(
  results_dir,
  sprintf(
    "average_kl_d_%s_lambda_%s_alpha_%s_beta_%s_opportunity_%s_expansion_%s_%dn.pdf",
    input_type, lambda_arg, alpha_arg, beta_label, opportunity_label, expansion_label, tree_size
  )
)

pdf(kl_pdf, width = 9.5, height = 5.5)
old_par <- par(mar = c(4.5, 4.5, 1, 8))
plot(
  NA,
  xlim = expand_range(kl_summary$timestep, pad = 0.1),
  ylim = expand_range(kl_summary$kl_d, pad = 0.05),
  xlab = "Timestep",
  ylab = "Average kl_d",
  main = "",
  xaxt = "n"
)
axis(1, at = sort(unique(kl_summary$timestep)))
grid()

for (opportunity_value in opportunity_levels) {
  for (beta_value in beta_levels) {
    beta_dat <- kl_summary[
      kl_summary$beta == beta_value &
        kl_summary$opportunity == opportunity_value,
      ,
      drop = FALSE
    ]
    beta_dat <- beta_dat[order(beta_dat$timestep), , drop = FALSE]
    if (nrow(beta_dat) > 0) {
      lines(
        beta_dat$timestep,
        beta_dat$kl_d,
        type = "b",
        pch = opportunity_pch[[opportunity_value]],
        lwd = 2,
        lty = opportunity_lty[[opportunity_value]],
        col = series_color(beta_value, opportunity_value)
      )
    }
  }
}

plot_parameter_legend()
par(old_par)
dev.off()

kl_reward_t1_pdf <- file.path(
  results_dir,
  sprintf(
    "kl_d_t1_by_reward_%s_lambda_%s_alpha_%s_beta_%s_opportunity_%s_expansion_%s_%dn.pdf",
    input_type, lambda_arg, alpha_arg, beta_label, opportunity_label, expansion_label, tree_size
  )
)

pdf(kl_reward_t1_pdf, width = 9.5, height = 5.5)
old_par <- par(mar = c(4.5, 4.5, 1, 8))
if (nrow(kl_reward_t1_summary) == 0) {
  plot(
    NA,
    xlim = c(-0.1, 1.1),
    ylim = c(0, 1),
    xlab = "Reward observed at timestep 1",
    ylab = "Average kl_d at timestep 1",
    main = "",
    xaxt = "n",
    yaxt = "n"
  )
  grid()
  text(0.5, 0.5, "No observed rewards at timestep 1", cex = 0.9)
} else {
  plot(
    NA,
    xlim = expand_range(kl_reward_t1_summary$reward, pad = 0.1),
    ylim = expand_range(kl_reward_t1_summary$kl_d, pad = 0.05),
    xlab = "Reward observed at timestep 1",
    ylab = "Average kl_d at timestep 1",
    main = "",
    xaxt = "n"
  )
  axis(1, at = sort(unique(kl_reward_t1_summary$reward)))
  grid()

  for (opportunity_value in opportunity_levels) {
    for (beta_value in beta_levels) {
      beta_dat <- kl_reward_t1_summary[
        kl_reward_t1_summary$beta == beta_value &
          kl_reward_t1_summary$opportunity == opportunity_value,
        ,
        drop = FALSE
      ]
      beta_dat <- beta_dat[order(beta_dat$reward), , drop = FALSE]
      if (nrow(beta_dat) > 0) {
        lines(
          beta_dat$reward,
          beta_dat$kl_d,
          type = "b",
          pch = opportunity_pch[[opportunity_value]],
          lwd = 2,
          lty = opportunity_lty[[opportunity_value]],
          col = series_color(beta_value, opportunity_value)
        )
      }
    }
  }
}
plot_parameter_legend()
par(old_par)
dev.off()

kl_reward_pdf <- file.path(
  results_dir,
  sprintf(
    "kl_d_by_reward_%s_lambda_%s_alpha_%s_beta_%s_opportunity_%s_expansion_%s_%dn.pdf",
    input_type, lambda_arg, alpha_arg, beta_label, opportunity_label, expansion_label, tree_size
  )
)

pdf(kl_reward_pdf, width = max(9.5, 4.2 * max(1, length(unique(kl_reward_summary$timestep))) + 3), height = 5.5)
plot_reward_timestep_summary(
  kl_reward_summary,
  value_col = "kl_d",
  ylab = "Average kl_d",
  main_prefix = "kl_d by observed reward",
  empty_message = "No observed rewards",
  y_limits = NULL
)
dev.off()

reconstruction_accuracy_pdf <- file.path(
  results_dir,
  sprintf(
    "reconstruction_accuracy_by_reward_%s_lambda_%s_alpha_%s_beta_%s_opportunity_%s_expansion_%s_%dn.pdf",
    input_type, lambda_arg, alpha_arg, beta_label, opportunity_label, expansion_label, tree_size
  )
)

pdf(
  reconstruction_accuracy_pdf,
  width = max(9.5, 4.2 * max(1, length(unique(reconstruction_accuracy_summary$timestep))) + 3),
  height = 5.5
)
plot_reward_timestep_summary(
  reconstruction_accuracy_summary,
  value_col = "accuracy",
  ylab = "Reconstruction accuracy",
  main_prefix = "Reconstruction accuracy by reward",
  empty_message = "No reconstructed rewards",
  y_limits = c(0, 1)
)
dev.off()

choice_stop_pdf <- file.path(
  results_dir,
  sprintf(
    "chosen_reward_given_stop_%s_lambda_%s_alpha_%s_beta_%s_opportunity_%s_expansion_%s_%dn.pdf",
    input_type, lambda_arg, alpha_arg, beta_label, opportunity_label, expansion_label, tree_size
  )
)

choice_stop_plot_data <- choice_stop_summary
names(choice_stop_plot_data)[names(choice_stop_plot_data) == "stop_timestep"] <- "timestep"
pdf(
  choice_stop_pdf,
  width = max(9.5, 4.2 * max(1, length(unique(choice_stop_plot_data$timestep))) + 3),
  height = 5.5
)
plot_reward_timestep_summary(
  choice_stop_plot_data,
  value_col = "p_choose",
  ylab = "P(chosen reward | stop)",
  main_prefix = "Chosen reward distribution given stop",
  empty_message = "No stopped trials",
  y_limits = c(0, 1)
)
dev.off()

choice_other_pdf <- file.path(
  results_dir,
  sprintf(
    "choice_probability_by_other_reward_stop_t2_%s_lambda_%s_alpha_%s_beta_%s_opportunity_%s_expansion_%s_%dn.pdf",
    input_type, lambda_arg, alpha_arg, beta_label, opportunity_label, expansion_label, tree_size
  )
)

plot_choice_other_panels <- function(summary_data) {
  other_rewards <- sort(unique(summary_data$other_reward))
  other_rewards <- other_rewards[!is.na(other_rewards)]
  if (length(other_rewards) == 0) {
    other_rewards <- NA_real_
  }

  n_cols <- min(4, length(other_rewards))
  n_rows <- ceiling(length(other_rewards) / n_cols)
  old_par <- par(mfrow = c(n_rows, n_cols), mar = c(4.5, 4.5, 1, 8))

  for (other_reward in other_rewards) {
    if (is.na(other_reward)) {
      plot(
        NA,
        xlim = c(-0.1, 1.1),
        ylim = c(0, 1),
        xlab = "Candidate reward",
        ylab = "P(choose candidate)",
        main = "Other reward: NA",
        xaxt = "n"
      )
      grid()
      text(0.5, 0.5, "No stopped-at-t2 choices", cex = 0.9)
      next
    }

    panel_data <- summary_data[summary_data$other_reward == other_reward, , drop = FALSE]
    plot(
      NA,
      xlim = expand_range(panel_data$reward, pad = 0.1),
      ylim = c(0, 1),
      xlab = "Candidate reward",
      ylab = "P(choose candidate)",
      main = sprintf("Other reward: %s", other_reward),
      xaxt = "n"
    )
    axis(1, at = sort(unique(panel_data$reward)))
    grid()

    for (opportunity_value in opportunity_levels) {
      for (beta_value in beta_levels) {
        series_data <- panel_data[
          panel_data$beta == beta_value &
            panel_data$opportunity == opportunity_value,
          ,
          drop = FALSE
        ]
        series_data <- series_data[order(series_data$reward), , drop = FALSE]
        if (nrow(series_data) > 0) {
          lines(
            series_data$reward,
            series_data$p_choose,
            type = "b",
            pch = opportunity_pch[[opportunity_value]],
            lwd = 2,
            lty = opportunity_lty[[opportunity_value]],
            col = series_color(beta_value, opportunity_value)
          )
        }
      }
    }
  }

  plot_parameter_legend()
  par(old_par)
}

pdf(
  choice_other_pdf,
  width = max(9.5, 3.5 * min(4, max(1, length(unique(choice_other_summary$other_reward)))) + 3),
  height = max(5.5, 3.8 * ceiling(max(1, length(unique(choice_other_summary$other_reward))) / 4))
)
plot_choice_other_panels(choice_other_summary)
dev.off()

message("Wrote: ", continue_pdf)
message("Wrote: ", v_mi_pdf)
message("Wrote: ", kl_pdf)
message("Wrote: ", kl_reward_t1_pdf)
message("Wrote: ", kl_reward_pdf)
message("Wrote: ", reconstruction_accuracy_pdf)
message("Wrote: ", choice_stop_pdf)
message("Wrote: ", choice_other_pdf)
