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
model_variant <- get_arg(10, "vae")
tree_config <- get_arg(11, "")

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

normalize_model_variant <- function(variant) {
  variant_key <- tolower(trimws(as.character(variant)))
  aliases <- c(
    "vae" = "vae",
    "autoencoder" = "vae",
    "rnn" = "rnn",
    "plain_rnn" = "rnn",
    "no_autoencoder" = "rnn",
    "no_ae" = "rnn"
  )
  if (!variant_key %in% names(aliases)) {
    stop(sprintf(
      "model_variant must be one of: %s. Got %s.",
      paste(names(aliases), collapse = ", "),
      variant
    ))
  }
  unname(aliases[[variant_key]])
}

model_variant <- normalize_model_variant(model_variant)

normalize_tree_config <- function(config) {
  key <- tolower(trimws(as.character(config)))
  if (!nzchar(key)) {
    return("")
  }
  aliases <- c(
    "auto" = "",
    "default" = "",
    "legacy" = "",
    "3armed" = "bandit3",
    "3_arm" = "bandit3",
    "3_armed" = "bandit3",
    "3-armed" = "bandit3",
    "3_arm_bandit" = "bandit3",
    "3-armed-bandit" = "bandit3",
    "three_arm_bandit" = "bandit3",
    "bandit3" = "bandit3",
    "4armed" = "bandit4",
    "4_arm" = "bandit4",
    "4_armed" = "bandit4",
    "4-armed" = "bandit4",
    "4_arm_bandit" = "bandit4",
    "4-armed-bandit" = "bandit4",
    "four_arm_bandit" = "bandit4",
    "bandit4" = "bandit4",
    "2x2" = "disjoint2x2",
    "2x2_disjoint" = "disjoint2x2",
    "disjoint2x2" = "disjoint2x2",
    "disjoint_2x2" = "disjoint2x2",
    "2path2node" = "disjoint2x2",
    "2_path_2_node" = "disjoint2x2",
    "2paths_2nodes" = "disjoint2x2",
    "2paths_2nodes_disjoint" = "disjoint2x2",
    "two_paths_two_nodes" = "disjoint2x2",
    "3x2" = "disjoint3x2",
    "3x2_disjoint" = "disjoint3x2",
    "disjoint3x2" = "disjoint3x2",
    "disjoint_3x2" = "disjoint3x2",
    "3path2node" = "disjoint3x2",
    "3_path_2_node" = "disjoint3x2",
    "3paths_2nodes" = "disjoint3x2",
    "3paths_2nodes_disjoint" = "disjoint3x2",
    "three_paths_two_nodes" = "disjoint3x2"
  )
  if (!key %in% names(aliases)) {
    stop(sprintf("Unknown tree_config=%s.", config))
  }
  unname(aliases[[key]])
}

tree_config <- normalize_tree_config(tree_config)
tree_file_label <- paste0(tree_size, "n", if (nzchar(tree_config)) paste0("_", tree_config) else "")

model_variant_file_segment <- function(variant) {
  sprintf("variant_%s_", variant)
}

model_variant_file_segments <- function(variant) {
  segments <- model_variant_file_segment(variant)
  if (identical(variant, "vae")) {
    # Backward compatibility for older VAE outputs that predated explicit
    # variant_vae_ filename segments.
    segments <- c(segments, "")
  }
  unique(segments)
}

dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)

beta_values <- trimws(strsplit(beta_arg, ",")[[1]])
opportunity_values <- trimws(strsplit(opportunity_arg, ",")[[1]])
seeds <-1:3

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
expansion_label <- sprintf("%s_variant_%s", expansion_label, arg_label(model_variant))
variant_file_segments <- model_variant_file_segments(model_variant)

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
    rounded_1 <- round(x_num, 1)
    rounded_2 <- round(x_num, 2)
    candidates <- c(
      candidates,
      format(x_num, scientific = FALSE, trim = TRUE)
    )
    if (abs(x_num - rounded_1) < 1e-12) {
      candidates <- c(candidates, sprintf("%.1f", rounded_1))
    }
    if (abs(x_num - rounded_2) < 1e-12) {
      candidates <- c(candidates, sprintf("%.2f", rounded_2))
    }
  }
  unique(candidates)
}

numeric_file_match <- function(lambda_value, alpha_value, beta_value, opportunity_value, seed) {
  requested <- suppressWarnings(as.numeric(c(
    lambda_value, alpha_value, beta_value, opportunity_value
  )))
  if (any(is.na(requested))) {
    return(NA_character_)
  }

  files <- list.files(input_dir, full.names = TRUE)
  for (variant_file_segment in variant_file_segments) {
    pattern <- paste0(
      "^lambda_([^_]+)_alpha_([^_]+)_beta_([^_]+)_opportunity_([^_]+)_",
      "expansion_", expansion_decision_version, "_", variant_file_segment,
      "seed_", seed, "_", tree_file_label, "_", input_type, "\\.csv$"
    )
    matches <- regexec(pattern, basename(files))
    pieces <- regmatches(basename(files), matches)
    for (i in seq_along(pieces)) {
      if (length(pieces[[i]]) == 0) {
        next
      }
      found <- suppressWarnings(as.numeric(pieces[[i]][2:5]))
      if (any(is.na(found))) {
        next
      }
      if (all(abs(found - requested) < 1e-8)) {
        return(files[[i]])
      }
    }
  }

  NA_character_
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
          for (variant_file_segment in variant_file_segments) {
            file_names <- c(
              sprintf(
                "lambda_%s_alpha_%s_beta_%s_opportunity_%s_expansion_%s_%sseed_%d_%s_%s.csv",
                lambda_candidate, alpha_candidate, beta_candidate, opportunity_candidate,
                expansion_decision_version, variant_file_segment, seed, tree_file_label, input_type
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
  }

  numeric_file_match(lambda_value, alpha_value, beta_value, opportunity_value, seed)
}

read_seed_file <- function(beta_value, opportunity_value, seed) {
  file_path <- simulation_path(lambda_arg, alpha_arg, beta_value, opportunity_value, seed)
  if (is.na(file_path)) {
    warning(sprintf(
      "Missing simulation file for beta=%s opportunity=%s seed=%d model_variant=%s",
      beta_value, opportunity_value, seed, model_variant
    ))
    return(NULL)
  }

  dat <- read.csv(file_path, stringsAsFactors = FALSE)
  dat <- drop_unnamed_index_columns(dat)
  dat$beta <- beta_value
  dat$opportunity <- opportunity_value
  dat$seed <- seed
  dat$model_variant <- model_variant
  dat$file_path <- file_path
  dat$source_file <- file_path
  dat
}

bind_rows_fill <- function(data_list) {
  if (length(data_list) == 0) {
    return(NULL)
  }

  all_cols <- unique(unlist(lapply(data_list, names), use.names = FALSE))
  aligned <- lapply(data_list, function(dat) {
    missing_cols <- setdiff(all_cols, names(dat))
    for (col in missing_cols) {
      dat[[col]] <- NA
    }
    dat[, all_cols, drop = FALSE]
  })

  do.call(rbind, aligned)
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
  bind_rows_fill(loaded_data)
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

trial_alive_before_decision <- function(row_data, decision_timestep) {
  if (decision_timestep <= 1) {
    return(TRUE)
  }
  for (previous_timestep in seq_len(decision_timestep - 1)) {
    stop_col <- paste0("stop_t", previous_timestep)
    if (!stop_col %in% names(row_data)) {
      next
    }
    stop_value <- row_data[[stop_col]][[1]]
    if (!is.na(stop_value) && as_logical_col(stop_value)) {
      return(FALSE)
    }
  }
  TRUE
}

column_timesteps <- function(dat, pattern, prefix_pattern) {
  cols <- grep(pattern, names(dat), value = TRUE)
  timesteps <- suppressWarnings(as.integer(sub(prefix_pattern, "", cols)))
  sort(timesteps[!is.na(timesteps)])
}

continue_reward_timesteps <- function(dat) {
  reward_timesteps <- column_timesteps(dat, "^expanded_reward_t[0-9]+$", "^expanded_reward_t")
  stop_timesteps <- column_timesteps(dat, "^stop_t[0-9]+$", "^stop_t")
  sort(reward_timesteps[(reward_timesteps + 1) %in% stop_timesteps])
}

observed_reward_timesteps <- function(dat) {
  column_timesteps(dat, "^expanded_reward_t[0-9]+$", "^expanded_reward_t")
}

kl_transition_timesteps <- function(dat) {
  reward_timesteps <- observed_reward_timesteps(dat)
  kl_timesteps <- vapply(kl_columns(dat), kl_timestep, integer(1))
  sort(kl_timesteps[
    kl_timesteps > 1 &
      kl_timesteps %in% reward_timesteps &
      (kl_timesteps - 1) %in% reward_timesteps
  ])
}

deep_probe_timesteps <- function(dat) {
  cols <- grep("^(lstm|decoder)_deep_probe_correct_t[0-9]+$", names(dat), value = TRUE)
  timesteps <- suppressWarnings(as.integer(sub("^(lstm|decoder)_deep_probe_correct_t", "", cols)))
  sort(unique(timesteps[!is.na(timesteps)]))
}

build_current_stop_data <- function(dat) {
  reward_cols <- grep("^expanded_reward_t[0-9]+$", names(dat), value = TRUE)
  reward_cols <- reward_cols[order(as.integer(sub("^expanded_reward_t", "", reward_cols)))]
  stop_cols <- grep("^stop_t[0-9]+$", names(dat), value = TRUE)
  stop_cols <- stop_cols[order(as.integer(sub("^stop_t", "", stop_cols)))]

  if (length(reward_cols) == 0 || length(stop_cols) == 0) {
    stop(
      paste(
        "Cannot compute timestep-specific continue probabilities.",
        "Expected expanded_reward_t* and stop_t* columns. The loaded files only contain:",
        paste(names(dat), collapse = ", ")
      )
    )
  }

  n_steps <- min(length(reward_cols), length(stop_cols))
  if (n_steps < 2) {
    stop("Need at least two timesteps to compute P(continue at t+1 | reward observed at t).")
  }

  trial_id_cols <- intersect(c("beta", "opportunity", "seed", "graph"), names(dat))
  trial_cols <- unique(c(trial_id_cols, reward_cols[seq_len(n_steps)], stop_cols[seq_len(n_steps)]))
  trial_data <- unique(dat[, trial_cols, drop = FALSE])

  rows <- list()
  row_i <- 1
  for (i in seq_len(nrow(trial_data))) {
    row_data <- trial_data[i, , drop = FALSE]
    for (t in seq_len(n_steps - 1)) {
      reward_t <- suppressWarnings(as.numeric(trial_data[[reward_cols[[t]]]][[i]]))
      stop_value <- trial_data[[stop_cols[[t + 1]]]][[i]]
      if (
        !is.na(reward_t) &&
          !is.na(stop_value) &&
          trial_alive_before_decision(row_data, t + 1)
      ) {
        rows[[row_i]] <- data.frame(
          beta = trial_data$beta[[i]],
          opportunity = trial_data$opportunity[[i]],
          seed = trial_data$seed[[i]],
          graph = trial_data$graph[[i]],
          reward_timestep = t,
          decision_timestep = t + 1,
          reward = reward_t,
          stop_current = as_logical_col(stop_value),
          stringsAsFactors = FALSE
        )
        row_i <- row_i + 1
      }
    }
  }

  if (length(rows) == 0) {
    warning("No observed rewards were found before a current decision; continue-by-reward panels will be empty.")
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

build_initial_stop_summary <- function(dat) {
  if (!"stop_t1" %in% names(dat)) {
    warning("No stop_t1 column was found; initial-stop panel will be empty.")
    return(data.frame(
      beta = character(),
      opportunity = character(),
      p_stop_initial = numeric(),
      p_continue_initial = numeric(),
      n = integer(),
      stringsAsFactors = FALSE
    ))
  }

  trial_id_cols <- intersect(c("beta", "opportunity", "seed", "graph", "stop_t1"), names(dat))
  trial_data <- unique(dat[, trial_id_cols, drop = FALSE])
  trial_data$stop_initial <- as_logical_col(trial_data$stop_t1)

  initial_summary <- aggregate(
    stop_initial ~ beta + opportunity,
    data = trial_data,
    FUN = mean
  )
  names(initial_summary)[names(initial_summary) == "stop_initial"] <- "p_stop_initial"
  initial_summary$p_continue_initial <- 1 - initial_summary$p_stop_initial

  initial_counts <- aggregate(
    stop_initial ~ beta + opportunity,
    data = trial_data,
    FUN = length
  )
  names(initial_counts)[names(initial_counts) == "stop_initial"] <- "n"

  merge(initial_summary, initial_counts, by = c("beta", "opportunity"))
}

initial_stop_summary <- build_initial_stop_summary(all_data)

build_t3_conditioned_continue_summary <- function(dat) {
  required_cols <- c("expanded_reward_t1", "expanded_reward_t2", "stop_t3")
  if (any(!required_cols %in% names(dat))) {
    warning(
      "Cannot build t3-conditioned continue heatmaps; expected expanded_reward_t1, expanded_reward_t2, and stop_t3 columns."
    )
    return(data.frame(
      beta = character(),
      opportunity = character(),
      reward_t1 = numeric(),
      reward_t2 = numeric(),
      p_continue_t3 = numeric(),
      n = integer(),
      stringsAsFactors = FALSE
    ))
  }

  trial_id_cols <- intersect(c("beta", "opportunity", "seed", "graph"), names(dat))
  prior_stop_cols <- intersect(c("stop_t1", "stop_t2"), names(dat))
  trial_cols <- unique(c(trial_id_cols, required_cols, prior_stop_cols))
  trial_data <- unique(dat[, trial_cols, drop = FALSE])
  trial_data$reward_t1 <- suppressWarnings(as.numeric(trial_data$expanded_reward_t1))
  trial_data$reward_t2 <- suppressWarnings(as.numeric(trial_data$expanded_reward_t2))
  stop_t3_raw <- trial_data$stop_t3
  trial_data$valid_stop_t3 <- !is.na(stop_t3_raw)
  trial_data$continue_t3 <- as.numeric(!as_logical_col(stop_t3_raw))
  trial_data$alive_t3 <- vapply(
    seq_len(nrow(trial_data)),
    function(row_i) trial_alive_before_decision(trial_data[row_i, , drop = FALSE], 3),
    logical(1)
  )
  trial_data <- trial_data[
    !is.na(trial_data$reward_t1) &
      !is.na(trial_data$reward_t2) &
      trial_data$valid_stop_t3 &
      trial_data$alive_t3,
    ,
    drop = FALSE
  ]

  if (nrow(trial_data) == 0) {
    warning("No trials had both reward_t1 and reward_t2 before a valid t3 decision; t3-conditioned continue heatmaps will be empty.")
    return(data.frame(
      beta = character(),
      opportunity = character(),
      reward_t1 = numeric(),
      reward_t2 = numeric(),
      p_continue_t3 = numeric(),
      n = integer(),
      stringsAsFactors = FALSE
    ))
  }

  t3_summary <- aggregate(
    continue_t3 ~ beta + opportunity + reward_t1 + reward_t2,
    data = trial_data,
    FUN = mean
  )
  names(t3_summary)[names(t3_summary) == "continue_t3"] <- "p_continue_t3"

  t3_counts <- aggregate(
    continue_t3 ~ beta + opportunity + reward_t1 + reward_t2,
    data = trial_data,
    FUN = length
  )
  names(t3_counts)[names(t3_counts) == "continue_t3"] <- "n"

  merge(
    t3_summary,
    t3_counts,
    by = c("beta", "opportunity", "reward_t1", "reward_t2")
  )
}

is_bandit3 <- identical(tree_config, "bandit3") || tree_size == 3
is_bandit4 <- identical(tree_config, "bandit4")
is_disjoint2x2 <- identical(tree_config, "disjoint2x2")
is_disjoint3x2 <- identical(tree_config, "disjoint3x2")
is_disjoint_tree <- is_disjoint2x2 || is_disjoint3x2
is_legacy6_tree <- tree_size == 6 && !nzchar(tree_config) && !is_disjoint_tree
disjoint_path_count <- if (is_disjoint3x2) {
  3L
} else if (is_disjoint2x2) {
  2L
} else {
  NA_integer_
}
disjoint_nodes_per_path <- 2L
has_t3_conditioned_continue <- is_bandit3 || is_bandit4
continue_t3_conditioned_summary <- if (has_t3_conditioned_continue) {
  build_t3_conditioned_continue_summary(all_data)
} else {
  data.frame(
    beta = character(),
    opportunity = character(),
    reward_t1 = numeric(),
    reward_t2 = numeric(),
    p_continue_t3 = numeric(),
    n = integer(),
    stringsAsFactors = FALSE
  )
}

disjoint_path_id <- function(node) {
  node_num <- suppressWarnings(as.numeric(node))
  out <- rep(NA_real_, length(node_num))
  valid <- !is.na(node_num)
  out[valid] <- floor((node_num[valid] - 1) / 2)
  out
}

empty_disjoint3x2_t1_path_action_summary <- function() {
  data.frame(
    beta = character(),
    opportunity = character(),
    reward_t1 = numeric(),
    p_continue_current_path = numeric(),
    p_continue_different_path = numeric(),
    p_stop = numeric(),
    n = integer(),
    stringsAsFactors = FALSE
  )
}

empty_disjoint3x2_later_path_action_summary <- function(include_complete = FALSE) {
  out <- data.frame(
    beta = character(),
    opportunity = character(),
    decision_timestep = integer(),
    best_path_value = numeric(),
    p_stop = numeric(),
    n = integer(),
    stringsAsFactors = FALSE
  )
  if (include_complete) {
    out$p_continue <- numeric()
    out <- out[, c("beta", "opportunity", "decision_timestep", "best_path_value", "p_continue", "p_stop", "n"), drop = FALSE]
  } else {
    out$p_continue_current_path <- numeric()
    out$p_continue_different_path <- numeric()
    out <- out[, c("beta", "opportunity", "decision_timestep", "best_path_value", "p_continue_current_path", "p_continue_different_path", "p_stop", "n"), drop = FALSE]
  }
  out
}

empty_disjoint3x2_best_path_continue_summary <- function() {
  data.frame(
    beta = character(),
    opportunity = character(),
    decision_timestep = integer(),
    best_path_value = numeric(),
    p_continue = numeric(),
    n = integer(),
    stringsAsFactors = FALSE
  )
}

empty_disjoint_ever_second_node_summary <- function() {
  data.frame(
    beta = character(),
    opportunity = character(),
    first_reward_on_path = numeric(),
    p_ever_second_node = numeric(),
    n = integer(),
    stringsAsFactors = FALSE
  )
}

legacy6_paths <- list(
  c(1, 2),
  c(1, 3),
  c(4, 5),
  c(4, 6)
)

legacy6_node_paths <- function(node) {
  node_value <- suppressWarnings(as.integer(node))
  if (length(node_value) == 0 || is.na(node_value)) {
    return(integer())
  }
  which(vapply(legacy6_paths, function(path_nodes) node_value %in% path_nodes, logical(1)))
}

legacy6_node_type <- function(node) {
  node_value <- suppressWarnings(as.integer(node))
  if (is.na(node_value)) {
    return(NA_character_)
  }
  if (node_value %in% c(1L, 4L)) {
    return("middle")
  }
  if (node_value %in% c(2L, 3L, 5L, 6L)) {
    return("leaf")
  }
  NA_character_
}

legacy6_same_path <- function(node_a, node_b) {
  paths_a <- legacy6_node_paths(node_a)
  paths_b <- legacy6_node_paths(node_b)
  length(intersect(paths_a, paths_b)) > 0
}

legacy6_path_state_before_decision <- function(row_data, decision_timestep) {
  path_values <- rep(0, length(legacy6_paths))
  path_counts <- rep(0L, length(legacy6_paths))
  observed_path_nodes <- vector("list", length(legacy6_paths))
  if (decision_timestep <= 1) {
    return(list(values = path_values, counts = path_counts))
  }

  for (observed_timestep in seq_len(decision_timestep - 1)) {
    node_col <- paste0("expanded_node_t", observed_timestep)
    reward_col <- paste0("expanded_reward_t", observed_timestep)
    if (!node_col %in% names(row_data) || !reward_col %in% names(row_data)) {
      next
    }
    node_value <- suppressWarnings(as.integer(row_data[[node_col]][[1]]))
    reward_value <- suppressWarnings(as.numeric(row_data[[reward_col]][[1]]))
    if (is.na(node_value) || is.na(reward_value)) {
      next
    }

    for (path_index in legacy6_node_paths(node_value)) {
      if (node_value %in% observed_path_nodes[[path_index]]) {
        next
      }
      observed_path_nodes[[path_index]] <- c(observed_path_nodes[[path_index]], node_value)
      path_values[[path_index]] <- path_values[[path_index]] + reward_value
      path_counts[[path_index]] <- path_counts[[path_index]] + 1L
    }
  }
  list(values = path_values, counts = path_counts)
}

empty_legacy6_continue_node_summary <- function() {
  data.frame(
    beta = character(),
    opportunity = character(),
    decision_timestep = integer(),
    node_index = integer(),
    p_selected_given_continue = numeric(),
    n = integer(),
    stringsAsFactors = FALSE
  )
}

empty_legacy6_path_action_by_reward_summary <- function() {
  data.frame(
    beta = character(),
    opportunity = character(),
    decision_timestep = integer(),
    observed_timestep = integer(),
    observed_node_type = character(),
    observed_reward = numeric(),
    p_continue_current_path = numeric(),
    p_continue_different_path = numeric(),
    p_stop = numeric(),
    n = integer(),
    stringsAsFactors = FALSE
  )
}

empty_legacy6_value_continue_summary <- function() {
  data.frame(
    beta = character(),
    opportunity = character(),
    decision_timestep = integer(),
    path_value = numeric(),
    p_continue = numeric(),
    n = integer(),
    stringsAsFactors = FALSE
  )
}

unique_trial_rows <- function(dat, required_cols) {
  trial_id_cols <- intersect(c("beta", "opportunity", "seed", "graph"), names(dat))
  trial_cols <- unique(c(trial_id_cols, required_cols))
  unique(dat[, trial_cols, drop = FALSE])
}

summarize_probability_columns <- function(dat, group_cols, value_cols, rename_map = NULL) {
  if (nrow(dat) == 0) {
    return(dat)
  }
  summary_data <- aggregate(
    dat[, value_cols, drop = FALSE],
    dat[, group_cols, drop = FALSE],
    FUN = mean
  )
  count_data <- aggregate(
    dat[[value_cols[[1]]]],
    dat[, group_cols, drop = FALSE],
    FUN = length
  )
  names(count_data)[ncol(count_data)] <- "n"
  out <- merge(summary_data, count_data, by = group_cols)
  if (!is.null(rename_map)) {
    for (old_name in names(rename_map)) {
      if (old_name %in% names(out)) {
        names(out)[names(out) == old_name] <- rename_map[[old_name]]
      }
    }
  }
  out
}

build_legacy6_continue_node_summary <- function(dat) {
  decision_timesteps <- column_timesteps(dat, "^stop_t[0-9]+$", "^stop_t")
  decision_timesteps <- decision_timesteps[paste0("expanded_node_t", decision_timesteps) %in% names(dat)]
  if (length(decision_timesteps) == 0) {
    return(empty_legacy6_continue_node_summary())
  }
  required_cols <- unique(c(
    paste0("expanded_node_t", decision_timesteps),
    paste0("stop_t", decision_timesteps)
  ))
  trial_data <- unique_trial_rows(dat, required_cols)
  rows <- list()

  for (decision_timestep in decision_timesteps) {
    stop_col <- paste0("stop_t", decision_timestep)
    node_col <- paste0("expanded_node_t", decision_timestep)
    for (row_i in seq_len(nrow(trial_data))) {
      row_data <- trial_data[row_i, , drop = FALSE]
      if (is.na(row_data[[stop_col]][[1]]) || !trial_alive_before_decision(row_data, decision_timestep)) {
        next
      }
      if (as_logical_col(row_data[[stop_col]][[1]])) {
        next
      }
      selected_node <- suppressWarnings(as.integer(row_data[[node_col]][[1]]))
      if (is.na(selected_node)) {
        next
      }
      for (node_index in seq_len(6)) {
        rows[[length(rows) + 1]] <- data.frame(
          beta = trial_data$beta[[row_i]],
          opportunity = trial_data$opportunity[[row_i]],
          decision_timestep = decision_timestep,
          node_index = node_index,
          selected_node_action = as.numeric(selected_node == node_index),
          stringsAsFactors = FALSE
        )
      }
    }
  }

  if (length(rows) == 0) {
    return(empty_legacy6_continue_node_summary())
  }
  node_data <- do.call(rbind, rows)
  summarize_probability_columns(
    node_data,
    group_cols = c("beta", "opportunity", "decision_timestep", "node_index"),
    value_cols = c("selected_node_action"),
    rename_map = c(selected_node_action = "p_selected_given_continue")
  )
}

build_legacy6_path_action_by_reward_summary <- function(dat) {
  reward_timesteps <- continue_reward_timesteps(dat)
  if (length(reward_timesteps) == 0) {
    return(empty_legacy6_path_action_by_reward_summary())
  }
  decision_timesteps <- reward_timesteps + 1L
  required_cols <- unique(c(
    paste0("expanded_node_t", reward_timesteps),
    paste0("expanded_reward_t", reward_timesteps),
    paste0("expanded_node_t", decision_timesteps),
    paste0("stop_t", decision_timesteps)
  ))
  required_cols <- required_cols[required_cols %in% names(dat)]
  trial_data <- unique_trial_rows(dat, required_cols)
  rows <- list()

  for (observed_timestep in reward_timesteps) {
    decision_timestep <- observed_timestep + 1L
    stop_col <- paste0("stop_t", decision_timestep)
    current_node_col <- paste0("expanded_node_t", observed_timestep)
    reward_col <- paste0("expanded_reward_t", observed_timestep)
    next_node_col <- paste0("expanded_node_t", decision_timestep)
    if (!stop_col %in% names(trial_data) || !current_node_col %in% names(trial_data)) {
      next
    }

    for (row_i in seq_len(nrow(trial_data))) {
      row_data <- trial_data[row_i, , drop = FALSE]
      if (is.na(row_data[[stop_col]][[1]]) || !trial_alive_before_decision(row_data, decision_timestep)) {
        next
      }
      current_node <- suppressWarnings(as.integer(row_data[[current_node_col]][[1]]))
      observed_reward <- suppressWarnings(as.numeric(row_data[[reward_col]][[1]]))
      node_type <- legacy6_node_type(current_node)
      if (is.na(current_node) || is.na(observed_reward) || is.na(node_type)) {
        next
      }
      stopped <- as_logical_col(row_data[[stop_col]][[1]])
      next_node <- if (!stopped && next_node_col %in% names(row_data)) {
        suppressWarnings(as.integer(row_data[[next_node_col]][[1]]))
      } else {
        NA_integer_
      }
      rows[[length(rows) + 1]] <- data.frame(
        beta = trial_data$beta[[row_i]],
        opportunity = trial_data$opportunity[[row_i]],
        decision_timestep = decision_timestep,
        observed_timestep = observed_timestep,
        observed_node_type = node_type,
        observed_reward = observed_reward,
        continue_current_path_action = as.numeric(
          !stopped && !is.na(next_node) && legacy6_same_path(current_node, next_node)
        ),
        continue_different_path_action = as.numeric(
          !stopped && !is.na(next_node) && !legacy6_same_path(current_node, next_node)
        ),
        stop_action = as.numeric(stopped),
        stringsAsFactors = FALSE
      )
    }
  }

  if (length(rows) == 0) {
    return(empty_legacy6_path_action_by_reward_summary())
  }
  action_data <- do.call(rbind, rows)
  summarize_probability_columns(
    action_data,
    group_cols = c("beta", "opportunity", "decision_timestep", "observed_timestep", "observed_node_type", "observed_reward"),
    value_cols = c("continue_current_path_action", "continue_different_path_action", "stop_action"),
    rename_map = c(
      continue_current_path_action = "p_continue_current_path",
      continue_different_path_action = "p_continue_different_path",
      stop_action = "p_stop"
    )
  )
}

build_legacy6_value_continue_summaries <- function(dat) {
  decision_timesteps <- continue_reward_timesteps(dat) + 1L
  decision_timesteps <- decision_timesteps[paste0("stop_t", decision_timesteps) %in% names(dat)]
  if (length(decision_timesteps) == 0) {
    return(list(
      complete = empty_legacy6_value_continue_summary(),
      best_observed = empty_legacy6_value_continue_summary()
    ))
  }
  required_cols <- unique(c(
    paste0("expanded_node_t", seq_len(max(decision_timesteps) - 1L)),
    paste0("expanded_reward_t", seq_len(max(decision_timesteps) - 1L)),
    paste0("stop_t", decision_timesteps)
  ))
  required_cols <- required_cols[required_cols %in% names(dat)]
  trial_data <- unique_trial_rows(dat, required_cols)
  complete_rows <- list()
  best_rows <- list()

  for (decision_timestep in decision_timesteps) {
    stop_col <- paste0("stop_t", decision_timestep)
    for (row_i in seq_len(nrow(trial_data))) {
      row_data <- trial_data[row_i, , drop = FALSE]
      if (is.na(row_data[[stop_col]][[1]]) || !trial_alive_before_decision(row_data, decision_timestep)) {
        next
      }
      state <- legacy6_path_state_before_decision(row_data, decision_timestep)
      continue_action <- as.numeric(!as_logical_col(row_data[[stop_col]][[1]]))

      observed_values <- state$values[state$counts > 0]
      if (length(observed_values) > 0) {
        best_rows[[length(best_rows) + 1]] <- data.frame(
          beta = trial_data$beta[[row_i]],
          opportunity = trial_data$opportunity[[row_i]],
          decision_timestep = decision_timestep,
          path_value = max(observed_values),
          continue_action = continue_action,
          stringsAsFactors = FALSE
        )
      }

      complete_values <- state$values[state$counts >= 2L]
      if (length(complete_values) > 0) {
        complete_rows[[length(complete_rows) + 1]] <- data.frame(
          beta = trial_data$beta[[row_i]],
          opportunity = trial_data$opportunity[[row_i]],
          decision_timestep = decision_timestep,
          path_value = max(complete_values),
          continue_action = continue_action,
          stringsAsFactors = FALSE
        )
      }
    }
  }

  summarize_or_empty <- function(rows) {
    if (length(rows) == 0) {
      return(empty_legacy6_value_continue_summary())
    }
    value_data <- do.call(rbind, rows)
    summarize_probability_columns(
      value_data,
      group_cols = c("beta", "opportunity", "decision_timestep", "path_value"),
      value_cols = c("continue_action"),
      rename_map = c(continue_action = "p_continue")
    )
  }

  list(
    complete = summarize_or_empty(complete_rows),
    best_observed = summarize_or_empty(best_rows)
  )
}

path_state_before_decision <- function(row_data, decision_timestep, n_paths) {
  path_values <- rep(0, n_paths)
  path_counts <- rep(0L, n_paths)
  for (observed_timestep in seq_len(decision_timestep - 1)) {
    node_col <- paste0("expanded_node_t", observed_timestep)
    reward_col <- paste0("expanded_reward_t", observed_timestep)
    if (!node_col %in% names(row_data) || !reward_col %in% names(row_data)) {
      next
    }
    node_value <- suppressWarnings(as.numeric(row_data[[node_col]][[1]]))
    reward_value <- suppressWarnings(as.numeric(row_data[[reward_col]][[1]]))
    path_value <- disjoint_path_id(node_value)
    if (!is.na(path_value) && !is.na(reward_value)) {
      path_index <- as.integer(path_value) + 1L
      if (path_index >= 1L && path_index <= n_paths) {
        path_values[[path_index]] <- path_values[[path_index]] + reward_value
        path_counts[[path_index]] <- path_counts[[path_index]] + 1L
      }
    }
  }
  list(values = path_values, counts = path_counts)
}

build_disjoint3x2_t1_path_action_summary <- function(dat) {
  required_cols <- c("expanded_node_t1", "expanded_reward_t1", "expanded_node_t2", "stop_t2")
  if (any(!required_cols %in% names(dat))) {
    warning("Cannot build disjoint3x2 t1 path-action plot; expected expanded_node/reward_t1, expanded_node_t2, and stop_t2 columns.")
    return(empty_disjoint3x2_t1_path_action_summary())
  }

  trial_data <- unique_trial_rows(dat, required_cols)
  trial_data$reward_t1 <- suppressWarnings(as.numeric(trial_data$expanded_reward_t1))
  trial_data$path_t1 <- disjoint_path_id(trial_data$expanded_node_t1)
  trial_data$path_t2 <- disjoint_path_id(trial_data$expanded_node_t2)
  trial_data$stop_action <- as.numeric(as_logical_col(trial_data$stop_t2))
  trial_data$continue_action <- as.numeric(!as_logical_col(trial_data$stop_t2))
  trial_data$current_path_action <- as.numeric(
    trial_data$continue_action == 1 &
      !is.na(trial_data$path_t1) &
      !is.na(trial_data$path_t2) &
      trial_data$path_t1 == trial_data$path_t2
  )
  trial_data$different_path_action <- as.numeric(
    trial_data$continue_action == 1 &
      !is.na(trial_data$path_t1) &
      !is.na(trial_data$path_t2) &
      trial_data$path_t1 != trial_data$path_t2
  )
  trial_data <- trial_data[
    !is.na(trial_data$reward_t1) &
      !is.na(trial_data$path_t1) &
      !is.na(trial_data$stop_t2),
    ,
    drop = FALSE
  ]
  if (nrow(trial_data) == 0) {
    warning("No valid disjoint3x2 trials had a t2 decision after observing t1.")
    return(empty_disjoint3x2_t1_path_action_summary())
  }

  summarize_probability_columns(
    trial_data,
    group_cols = c("beta", "opportunity", "reward_t1"),
    value_cols = c("current_path_action", "different_path_action", "stop_action"),
    rename_map = c(
      current_path_action = "p_continue_current_path",
      different_path_action = "p_continue_different_path",
      stop_action = "p_stop"
    )
  )
}

build_disjoint3x2_later_path_action_summaries <- function(dat) {
  max_decision_timestep <- min(tree_size, disjoint_path_count * disjoint_nodes_per_path)
  if (is.na(max_decision_timestep) || max_decision_timestep < 3) {
    return(list(
      incomplete = empty_disjoint3x2_later_path_action_summary(FALSE),
      complete = empty_disjoint3x2_later_path_action_summary(TRUE)
    ))
  }
  required_cols <- unique(c(
    paste0("expanded_node_t", seq_len(max_decision_timestep)),
    paste0("expanded_reward_t", seq_len(max_decision_timestep - 1)),
    paste0("stop_t", 3:max_decision_timestep)
  ))
  required_cols <- required_cols[required_cols %in% names(dat)]
  trial_data <- unique_trial_rows(dat, required_cols)
  incomplete_rows <- list()
  complete_rows <- list()

  for (decision_timestep in 3:max_decision_timestep) {
    stop_col <- paste0("stop_t", decision_timestep)
    action_node_col <- paste0("expanded_node_t", decision_timestep)
    current_node_col <- paste0("expanded_node_t", decision_timestep - 1)
    if (!stop_col %in% names(trial_data) || !current_node_col %in% names(trial_data)) {
      next
    }

    decision_rows <- list()
    for (row_i in seq_len(nrow(trial_data))) {
      if (is.na(trial_data[[stop_col]][[row_i]])) {
        next
      }
      row_data <- trial_data[row_i, , drop = FALSE]
      if (!trial_alive_before_decision(row_data, decision_timestep)) {
        next
      }
      current_path <- disjoint_path_id(trial_data[[current_node_col]][[row_i]])
      if (is.na(current_path)) {
        next
      }
      state <- path_state_before_decision(row_data, decision_timestep, disjoint_path_count)
      best_path_value <- max(state$values)
      best_path_observed <- any(
        abs(state$values - best_path_value) < 1e-12 &
          state$counts > 0
      )
      best_path_complete <- any(
        abs(state$values - best_path_value) < 1e-12 &
          state$counts >= disjoint_nodes_per_path
      )
      stopped <- as_logical_col(trial_data[[stop_col]][[row_i]])
      action_path <- if (!stopped && action_node_col %in% names(trial_data)) {
        disjoint_path_id(trial_data[[action_node_col]][[row_i]])
      } else {
        NA_real_
      }
      continue_action <- as.numeric(!stopped)
      current_path_action <- as.numeric(
        continue_action == 1 &&
          !is.na(action_path) &&
          action_path == current_path
      )
      different_path_action <- as.numeric(
        continue_action == 1 &&
          !is.na(action_path) &&
          action_path != current_path
      )
      decision_rows[[length(decision_rows) + 1]] <- data.frame(
        beta = trial_data$beta[[row_i]],
        opportunity = trial_data$opportunity[[row_i]],
        decision_timestep = decision_timestep,
        best_path_value = best_path_value,
        best_path_observed = best_path_observed,
        best_path_complete = best_path_complete,
        current_path_action = current_path_action,
        different_path_action = different_path_action,
        continue_action = continue_action,
        stop_action = as.numeric(stopped),
        stringsAsFactors = FALSE
      )
    }
    if (length(decision_rows) == 0) {
      next
    }
    decision_data <- do.call(rbind, decision_rows)
    incomplete_rows[[length(incomplete_rows) + 1]] <- decision_data[
      !decision_data$best_path_complete &
        decision_data$best_path_observed,
      ,
      drop = FALSE
    ]
    complete_rows[[length(complete_rows) + 1]] <- decision_data[
      decision_data$best_path_complete,
      ,
      drop = FALSE
    ]
  }

  incomplete_data <- if (length(incomplete_rows) > 0) do.call(rbind, incomplete_rows) else NULL
  complete_data <- if (length(complete_rows) > 0) do.call(rbind, complete_rows) else NULL

  incomplete_summary <- if (!is.null(incomplete_data) && nrow(incomplete_data) > 0) {
    summarize_probability_columns(
      incomplete_data,
      group_cols = c("beta", "opportunity", "decision_timestep", "best_path_value"),
      value_cols = c("current_path_action", "different_path_action", "stop_action"),
      rename_map = c(
        current_path_action = "p_continue_current_path",
        different_path_action = "p_continue_different_path",
        stop_action = "p_stop"
      )
    )
  } else {
    empty_disjoint3x2_later_path_action_summary(FALSE)
  }

  complete_summary <- if (!is.null(complete_data) && nrow(complete_data) > 0) {
    summarize_probability_columns(
      complete_data,
      group_cols = c("beta", "opportunity", "decision_timestep", "best_path_value"),
      value_cols = c("continue_action", "stop_action"),
      rename_map = c(
        continue_action = "p_continue",
        stop_action = "p_stop"
      )
    )
  } else {
    empty_disjoint3x2_later_path_action_summary(TRUE)
  }

  list(incomplete = incomplete_summary, complete = complete_summary)
}

build_disjoint3x2_best_path_continue_summary <- function(dat) {
  max_decision_timestep <- min(tree_size, disjoint_path_count * disjoint_nodes_per_path)
  if (is.na(max_decision_timestep) || max_decision_timestep < 2) {
    return(empty_disjoint3x2_best_path_continue_summary())
  }

  required_cols <- unique(c(
    paste0("expanded_node_t", seq_len(max_decision_timestep - 1)),
    paste0("expanded_reward_t", seq_len(max_decision_timestep - 1)),
    paste0("stop_t", 2:max_decision_timestep)
  ))
  required_cols <- required_cols[required_cols %in% names(dat)]
  trial_data <- unique_trial_rows(dat, required_cols)
  continue_rows <- list()

  for (decision_timestep in 2:max_decision_timestep) {
    stop_col <- paste0("stop_t", decision_timestep)
    if (!stop_col %in% names(trial_data)) {
      next
    }

    decision_rows <- list()
    for (row_i in seq_len(nrow(trial_data))) {
      if (is.na(trial_data[[stop_col]][[row_i]])) {
        next
      }

      row_data <- trial_data[row_i, , drop = FALSE]
      if (!trial_alive_before_decision(row_data, decision_timestep)) {
        next
      }
      state <- path_state_before_decision(row_data, decision_timestep, disjoint_path_count)
      observed_values <- state$values[state$counts > 0]
      if (length(observed_values) == 0) {
        next
      }

      decision_rows[[length(decision_rows) + 1]] <- data.frame(
        beta = trial_data$beta[[row_i]],
        opportunity = trial_data$opportunity[[row_i]],
        decision_timestep = decision_timestep,
        best_path_value = max(observed_values),
        continue_action = as.numeric(!as_logical_col(trial_data[[stop_col]][[row_i]])),
        stringsAsFactors = FALSE
      )
    }

    if (length(decision_rows) > 0) {
      continue_rows[[length(continue_rows) + 1]] <- do.call(rbind, decision_rows)
    }
  }

  if (length(continue_rows) == 0) {
    warning("No disjoint3x2 decisions had observed path values for the best-path continue plot.")
    return(empty_disjoint3x2_best_path_continue_summary())
  }

  continue_data <- do.call(rbind, continue_rows)
  summarize_probability_columns(
    continue_data,
    group_cols = c("beta", "opportunity", "decision_timestep", "best_path_value"),
    value_cols = c("continue_action"),
    rename_map = c(continue_action = "p_continue")
  )[, c("beta", "opportunity", "decision_timestep", "best_path_value", "p_continue", "n"), drop = FALSE]
}

build_disjoint_ever_second_node_summary <- function(dat) {
  if (!is_disjoint_tree || is.na(disjoint_path_count)) {
    return(empty_disjoint_ever_second_node_summary())
  }
  observed_timesteps <- observed_reward_timesteps(dat)
  if (length(observed_timesteps) == 0) {
    return(empty_disjoint_ever_second_node_summary())
  }
  required_cols <- unique(c(
    paste0("expanded_node_t", observed_timesteps),
    paste0("expanded_reward_t", observed_timesteps)
  ))
  if (any(!required_cols %in% names(dat))) {
    return(empty_disjoint_ever_second_node_summary())
  }
  trial_data <- unique_trial_rows(dat, required_cols)
  rows <- list()
  for (row_i in seq_len(nrow(trial_data))) {
    observed_nodes <- suppressWarnings(as.numeric(trial_data[row_i, paste0("expanded_node_t", observed_timesteps), drop = TRUE]))
    observed_rewards <- suppressWarnings(as.numeric(trial_data[row_i, paste0("expanded_reward_t", observed_timesteps), drop = TRUE]))
    observed_paths <- disjoint_path_id(observed_nodes)
    for (path_i in seq_len(disjoint_path_count) - 1L) {
      path_timesteps <- which(!is.na(observed_paths) & observed_paths == path_i & !is.na(observed_rewards))
      if (length(path_timesteps) == 0) {
        next
      }
      rows[[length(rows) + 1]] <- data.frame(
        beta = trial_data$beta[[row_i]],
        opportunity = trial_data$opportunity[[row_i]],
        first_reward_on_path = observed_rewards[[path_timesteps[[1]]]],
        ever_second_node = as.numeric(length(path_timesteps) >= 2),
        stringsAsFactors = FALSE
      )
    }
  }
  if (length(rows) == 0) {
    return(empty_disjoint_ever_second_node_summary())
  }
  event_data <- do.call(rbind, rows)
  summarize_probability_columns(
    event_data,
    group_cols = c("beta", "opportunity", "first_reward_on_path"),
    value_cols = c("ever_second_node"),
    rename_map = c(ever_second_node = "p_ever_second_node")
  )
}

build_disjoint2x2_t2_path_continue_summary <- function(dat) {
  required_cols <- c("expanded_node_t1", "expanded_reward_t1", "expanded_node_t2", "stop_t2")
  if (any(!required_cols %in% names(dat))) {
    warning(
      "Cannot build disjoint2x2 t2 path-choice plot; expected expanded_node_t1, expanded_reward_t1, expanded_node_t2, and stop_t2 columns."
    )
    return(data.frame(
      beta = character(),
      opportunity = character(),
      reward_t1 = numeric(),
      p_current_path_t2 = numeric(),
      p_different_path_t2 = numeric(),
      n = integer(),
      stringsAsFactors = FALSE
    ))
  }

  trial_id_cols <- intersect(c("beta", "opportunity", "seed", "graph"), names(dat))
  trial_cols <- unique(c(trial_id_cols, required_cols))
  trial_data <- unique(dat[, trial_cols, drop = FALSE])
  trial_data$reward_t1 <- suppressWarnings(as.numeric(trial_data$expanded_reward_t1))
  trial_data$path_t1 <- disjoint_path_id(trial_data$expanded_node_t1)
  trial_data$path_t2 <- disjoint_path_id(trial_data$expanded_node_t2)
  trial_data$continue_t2 <- !as_logical_col(trial_data$stop_t2)
  trial_data$same_path_t2 <- trial_data$path_t2 == trial_data$path_t1
  trial_data <- trial_data[
    !is.na(trial_data$reward_t1) &
      !is.na(trial_data$path_t1) &
      !is.na(trial_data$path_t2) &
      !is.na(trial_data$stop_t2) &
      trial_data$continue_t2,
    ,
    drop = FALSE
  ]

  if (nrow(trial_data) == 0) {
    warning("No disjoint2x2 trials continued at t2 with valid t1/t2 nodes; t2 path-choice plot will be empty.")
    return(data.frame(
      beta = character(),
      opportunity = character(),
      reward_t1 = numeric(),
      p_current_path_t2 = numeric(),
      p_different_path_t2 = numeric(),
      n = integer(),
      stringsAsFactors = FALSE
    ))
  }

  t2_summary <- aggregate(
    same_path_t2 ~ beta + opportunity + reward_t1,
    data = trial_data,
    FUN = mean
  )
  names(t2_summary)[names(t2_summary) == "same_path_t2"] <- "p_current_path_t2"
  t2_summary$p_different_path_t2 <- 1 - t2_summary$p_current_path_t2

  t2_counts <- aggregate(
    same_path_t2 ~ beta + opportunity + reward_t1,
    data = trial_data,
    FUN = length
  )
  names(t2_counts)[names(t2_counts) == "same_path_t2"] <- "n"

  merge(t2_summary, t2_counts, by = c("beta", "opportunity", "reward_t1"))
}

build_disjoint2x2_t3_path_continue_summary <- function(dat) {
  required_cols <- c(
    "expanded_node_t1",
    "expanded_reward_t1",
    "expanded_node_t2",
    "expanded_reward_t2",
    "expanded_node_t3",
    "stop_t3"
  )
  if (any(!required_cols %in% names(dat))) {
    warning(
      "Cannot build disjoint2x2 t3 path-choice heatmaps; expected expanded_node_t1/t2/t3, expanded_reward_t1/t2, and stop_t3 columns."
    )
    return(data.frame(
      beta = character(),
      opportunity = character(),
      reward_t1 = numeric(),
      reward_t2 = numeric(),
      p_current_path_t3 = numeric(),
      p_different_path_t3 = numeric(),
      n = integer(),
      stringsAsFactors = FALSE
    ))
  }

  trial_id_cols <- intersect(c("beta", "opportunity", "seed", "graph"), names(dat))
  trial_cols <- unique(c(trial_id_cols, required_cols))
  trial_data <- unique(dat[, trial_cols, drop = FALSE])
  trial_data$reward_t1 <- suppressWarnings(as.numeric(trial_data$expanded_reward_t1))
  trial_data$reward_t2 <- suppressWarnings(as.numeric(trial_data$expanded_reward_t2))
  trial_data$path_t1 <- disjoint_path_id(trial_data$expanded_node_t1)
  trial_data$path_t2 <- disjoint_path_id(trial_data$expanded_node_t2)
  trial_data$path_t3 <- disjoint_path_id(trial_data$expanded_node_t3)
  trial_data$continue_t3 <- !as_logical_col(trial_data$stop_t3)
  trial_data$different_paths_t1_t2 <- trial_data$path_t1 != trial_data$path_t2
  trial_data$same_as_current_path_t3 <- trial_data$path_t3 == trial_data$path_t2
  trial_data <- trial_data[
    !is.na(trial_data$reward_t1) &
      !is.na(trial_data$reward_t2) &
      !is.na(trial_data$path_t1) &
      !is.na(trial_data$path_t2) &
      !is.na(trial_data$path_t3) &
      !is.na(trial_data$stop_t3) &
      trial_data$different_paths_t1_t2 &
      trial_data$continue_t3,
    ,
    drop = FALSE
  ]

  if (nrow(trial_data) == 0) {
    warning("No disjoint2x2 trials continued at t3 after observing different paths at t1/t2; t3 path-choice heatmaps will be empty.")
    return(data.frame(
      beta = character(),
      opportunity = character(),
      reward_t1 = numeric(),
      reward_t2 = numeric(),
      p_current_path_t3 = numeric(),
      p_different_path_t3 = numeric(),
      n = integer(),
      stringsAsFactors = FALSE
    ))
  }

  t3_summary <- aggregate(
    same_as_current_path_t3 ~ beta + opportunity + reward_t1 + reward_t2,
    data = trial_data,
    FUN = mean
  )
  names(t3_summary)[names(t3_summary) == "same_as_current_path_t3"] <- "p_current_path_t3"
  t3_summary$p_different_path_t3 <- 1 - t3_summary$p_current_path_t3

  t3_counts <- aggregate(
    same_as_current_path_t3 ~ beta + opportunity + reward_t1 + reward_t2,
    data = trial_data,
    FUN = length
  )
  names(t3_counts)[names(t3_counts) == "same_as_current_path_t3"] <- "n"

  merge(
    t3_summary,
    t3_counts,
    by = c("beta", "opportunity", "reward_t1", "reward_t2")
  )
}

disjoint2x2_t2_path_continue_summary <- if (is_disjoint2x2) {
  build_disjoint2x2_t2_path_continue_summary(all_data)
} else {
  data.frame(
    beta = character(),
    opportunity = character(),
    reward_t1 = numeric(),
    p_current_path_t2 = numeric(),
    p_different_path_t2 = numeric(),
    n = integer(),
    stringsAsFactors = FALSE
  )
}

disjoint2x2_t3_path_continue_summary <- if (is_disjoint2x2) {
  build_disjoint2x2_t3_path_continue_summary(all_data)
} else {
  data.frame(
    beta = character(),
    opportunity = character(),
    reward_t1 = numeric(),
    reward_t2 = numeric(),
    p_current_path_t3 = numeric(),
    p_different_path_t3 = numeric(),
    n = integer(),
    stringsAsFactors = FALSE
  )
}

empty_disjoint2x2_t3_same_path_reward_summary <- function() {
  data.frame(
    beta = character(),
    opportunity = character(),
    observed_path_reward = numeric(),
    p_continue_t3 = numeric(),
    n = integer(),
    stringsAsFactors = FALSE
  )
}

empty_disjoint2x2_t3_different_path_reward_summary <- function(value_col = "p_continue_t3") {
  out <- data.frame(
    beta = character(),
    opportunity = character(),
    reward_t1 = numeric(),
    reward_t2 = numeric(),
    n = integer(),
    stringsAsFactors = FALSE
  )
  out[[value_col]] <- numeric()
  out[, c("beta", "opportunity", "reward_t1", "reward_t2", value_col, "n"), drop = FALSE]
}

empty_disjoint2x2_t4_observed_path_reward_summary <- function() {
  data.frame(
    beta = character(),
    opportunity = character(),
    observed_path_reward = numeric(),
    p_continue_t4 = numeric(),
    n = integer(),
    stringsAsFactors = FALSE
  )
}

empty_disjoint2x2_path_value_diff_continue_summary <- function() {
  data.frame(
    beta = character(),
    opportunity = character(),
    decision_timestep = integer(),
    value_diff = numeric(),
    p_continue = numeric(),
    n = integer(),
    stringsAsFactors = FALSE
  )
}

build_disjoint2x2_path_value_diff_continue_summary <- function(dat) {
  max_timestep <- min(4, tree_size)
  required_cols <- c(
    paste0("expanded_node_t", seq_len(max_timestep - 1)),
    paste0("expanded_reward_t", seq_len(max_timestep - 1)),
    paste0("stop_t", 2:max_timestep)
  )
  if (any(!required_cols %in% names(dat))) {
    warning(
      "Cannot build disjoint2x2 path-value-difference continue plot; expected expanded_node/reward_t1..t3 and stop_t2..t4 columns."
    )
    return(empty_disjoint2x2_path_value_diff_continue_summary())
  }

  trial_id_cols <- intersect(c("beta", "opportunity", "seed", "graph"), names(dat))
  trial_cols <- unique(c(trial_id_cols, required_cols))
  trial_data <- unique(dat[, trial_cols, drop = FALSE])
  summary_rows <- list()

  for (decision_timestep in 2:max_timestep) {
    stop_col <- paste0("stop_t", decision_timestep)
    current_node_col <- paste0("expanded_node_t", decision_timestep - 1)
    current_reward_col <- paste0("expanded_reward_t", decision_timestep - 1)
    decision_data <- trial_data[
      !is.na(trial_data[[stop_col]]) &
        !is.na(trial_data[[current_node_col]]) &
        !is.na(trial_data[[current_reward_col]]),
      ,
      drop = FALSE
    ]
    if (nrow(decision_data) == 0) {
      next
    }

    decision_data$continue_decision <- as.numeric(!as_logical_col(decision_data[[stop_col]]))
    decision_data$value_diff <- vapply(seq_len(nrow(decision_data)), function(row_i) {
      path_values <- c(0, 0)
      for (observed_timestep in seq_len(decision_timestep - 1)) {
        node_value <- suppressWarnings(as.numeric(decision_data[[paste0("expanded_node_t", observed_timestep)]][[row_i]]))
        reward_value <- suppressWarnings(as.numeric(decision_data[[paste0("expanded_reward_t", observed_timestep)]][[row_i]]))
        path_value <- disjoint_path_id(node_value)
        if (!is.na(path_value) && !is.na(reward_value)) {
          path_values[[as.integer(path_value) + 1]] <- path_values[[as.integer(path_value) + 1]] + reward_value
        }
      }

      current_path <- disjoint_path_id(decision_data[[current_node_col]][[row_i]])
      if (is.na(current_path)) {
        return(NA_real_)
      }
      current_path <- as.integer(current_path) + 1
      other_path <- ifelse(current_path == 1, 2, 1)
      path_values[[current_path]] - path_values[[other_path]]
    }, numeric(1))
    decision_data$decision_timestep <- decision_timestep
    decision_data <- decision_data[
      !is.na(decision_data$value_diff),
      ,
      drop = FALSE
    ]
    if (nrow(decision_data) == 0) {
      next
    }

    summary_rows[[length(summary_rows) + 1]] <- decision_data[
      ,
      c("beta", "opportunity", "decision_timestep", "value_diff", "continue_decision"),
      drop = FALSE
    ]
  }

  if (length(summary_rows) == 0) {
    warning("No disjoint2x2 decisions had valid current-vs-other observed path values.")
    return(empty_disjoint2x2_path_value_diff_continue_summary())
  }

  summary_data <- do.call(rbind, summary_rows)
  continue_summary <- aggregate(
    continue_decision ~ beta + opportunity + decision_timestep + value_diff,
    data = summary_data,
    FUN = mean
  )
  names(continue_summary)[names(continue_summary) == "continue_decision"] <- "p_continue"

  continue_counts <- aggregate(
    continue_decision ~ beta + opportunity + decision_timestep + value_diff,
    data = summary_data,
    FUN = length
  )
  names(continue_counts)[names(continue_counts) == "continue_decision"] <- "n"

  merge(
    continue_summary,
    continue_counts,
    by = c("beta", "opportunity", "decision_timestep", "value_diff")
  )
}

build_disjoint2x2_t3_same_path_reward_summary <- function(dat) {
  required_cols <- c(
    "expanded_node_t1",
    "expanded_reward_t1",
    "expanded_node_t2",
    "expanded_reward_t2",
    "stop_t2",
    "stop_t3"
  )
  if (any(!required_cols %in% names(dat))) {
    warning(
      "Cannot build disjoint2x2 t3 same-path continue plot; expected expanded_node/reward_t1/t2 and stop_t2/t3 columns."
    )
    return(empty_disjoint2x2_t3_same_path_reward_summary())
  }

  trial_id_cols <- intersect(c("beta", "opportunity", "seed", "graph"), names(dat))
  trial_cols <- unique(c(trial_id_cols, required_cols))
  trial_data <- unique(dat[, trial_cols, drop = FALSE])
  trial_data$reward_t1 <- suppressWarnings(as.numeric(trial_data$expanded_reward_t1))
  trial_data$reward_t2 <- suppressWarnings(as.numeric(trial_data$expanded_reward_t2))
  trial_data$path_t1 <- disjoint_path_id(trial_data$expanded_node_t1)
  trial_data$path_t2 <- disjoint_path_id(trial_data$expanded_node_t2)
  trial_data$valid_stop_t2 <- !is.na(trial_data$stop_t2)
  trial_data$valid_stop_t3 <- !is.na(trial_data$stop_t3)
  trial_data$continued_to_t3 <- !as_logical_col(trial_data$stop_t2)
  trial_data$continue_t3 <- as.numeric(!as_logical_col(trial_data$stop_t3))
  trial_data$observed_path_reward <- trial_data$reward_t1 + trial_data$reward_t2
  trial_data <- trial_data[
    !is.na(trial_data$reward_t1) &
      !is.na(trial_data$reward_t2) &
      !is.na(trial_data$path_t1) &
      !is.na(trial_data$path_t2) &
      trial_data$path_t1 == trial_data$path_t2 &
      trial_data$valid_stop_t2 &
      trial_data$valid_stop_t3 &
      trial_data$continued_to_t3,
    ,
    drop = FALSE
  ]

  if (nrow(trial_data) == 0) {
    warning("No disjoint2x2 trials reached t3 after observing two rewards in the same path; t3 same-path continue plot will be empty.")
    return(empty_disjoint2x2_t3_same_path_reward_summary())
  }

  t3_summary <- aggregate(
    continue_t3 ~ beta + opportunity + observed_path_reward,
    data = trial_data,
    FUN = mean
  )
  names(t3_summary)[names(t3_summary) == "continue_t3"] <- "p_continue_t3"

  t3_counts <- aggregate(
    continue_t3 ~ beta + opportunity + observed_path_reward,
    data = trial_data,
    FUN = length
  )
  names(t3_counts)[names(t3_counts) == "continue_t3"] <- "n"

  merge(t3_summary, t3_counts, by = c("beta", "opportunity", "observed_path_reward"))
}

build_disjoint2x2_t3_different_path_reward_summary <- function(dat) {
  required_cols <- c(
    "expanded_node_t1",
    "expanded_reward_t1",
    "expanded_node_t2",
    "expanded_reward_t2",
    "stop_t2",
    "stop_t3"
  )
  if (any(!required_cols %in% names(dat))) {
    warning(
      "Cannot build disjoint2x2 t3 different-path continue heatmaps; expected expanded_node/reward_t1/t2 and stop_t2/t3 columns."
    )
    return(empty_disjoint2x2_t3_different_path_reward_summary("p_continue_t3"))
  }

  trial_id_cols <- intersect(c("beta", "opportunity", "seed", "graph"), names(dat))
  trial_cols <- unique(c(trial_id_cols, required_cols))
  trial_data <- unique(dat[, trial_cols, drop = FALSE])
  trial_data$reward_t1 <- suppressWarnings(as.numeric(trial_data$expanded_reward_t1))
  trial_data$reward_t2 <- suppressWarnings(as.numeric(trial_data$expanded_reward_t2))
  trial_data$path_t1 <- disjoint_path_id(trial_data$expanded_node_t1)
  trial_data$path_t2 <- disjoint_path_id(trial_data$expanded_node_t2)
  trial_data$valid_stop_t2 <- !is.na(trial_data$stop_t2)
  trial_data$valid_stop_t3 <- !is.na(trial_data$stop_t3)
  trial_data$continued_to_t3 <- !as_logical_col(trial_data$stop_t2)
  trial_data$continue_t3 <- as.numeric(!as_logical_col(trial_data$stop_t3))
  trial_data <- trial_data[
    !is.na(trial_data$reward_t1) &
      !is.na(trial_data$reward_t2) &
      !is.na(trial_data$path_t1) &
      !is.na(trial_data$path_t2) &
      trial_data$path_t1 != trial_data$path_t2 &
      trial_data$valid_stop_t2 &
      trial_data$valid_stop_t3 &
      trial_data$continued_to_t3,
    ,
    drop = FALSE
  ]

  if (nrow(trial_data) == 0) {
    warning("No disjoint2x2 trials reached t3 after observing different paths at t1/t2; t3 different-path continue heatmaps will be empty.")
    return(empty_disjoint2x2_t3_different_path_reward_summary("p_continue_t3"))
  }

  t3_summary <- aggregate(
    continue_t3 ~ beta + opportunity + reward_t1 + reward_t2,
    data = trial_data,
    FUN = mean
  )
  names(t3_summary)[names(t3_summary) == "continue_t3"] <- "p_continue_t3"

  t3_counts <- aggregate(
    continue_t3 ~ beta + opportunity + reward_t1 + reward_t2,
    data = trial_data,
    FUN = length
  )
  names(t3_counts)[names(t3_counts) == "continue_t3"] <- "n"

  merge(t3_summary, t3_counts, by = c("beta", "opportunity", "reward_t1", "reward_t2"))
}

build_disjoint2x2_t3_better_path_continue_summary <- function(dat) {
  required_cols <- c(
    "expanded_node_t1",
    "expanded_reward_t1",
    "expanded_node_t2",
    "expanded_reward_t2",
    "expanded_node_t3",
    "stop_t2",
    "stop_t3"
  )
  if (any(!required_cols %in% names(dat))) {
    warning(
      "Cannot build disjoint2x2 t3 better-path continue heatmaps; expected expanded_node_t1/t2/t3, expanded_reward_t1/t2, and stop_t2/t3 columns."
    )
    return(empty_disjoint2x2_t3_different_path_reward_summary("p_continue_better_path_t3"))
  }

  trial_id_cols <- intersect(c("beta", "opportunity", "seed", "graph"), names(dat))
  trial_cols <- unique(c(trial_id_cols, required_cols))
  trial_data <- unique(dat[, trial_cols, drop = FALSE])
  trial_data$reward_t1 <- suppressWarnings(as.numeric(trial_data$expanded_reward_t1))
  trial_data$reward_t2 <- suppressWarnings(as.numeric(trial_data$expanded_reward_t2))
  trial_data$path_t1 <- disjoint_path_id(trial_data$expanded_node_t1)
  trial_data$path_t2 <- disjoint_path_id(trial_data$expanded_node_t2)
  trial_data$path_t3 <- disjoint_path_id(trial_data$expanded_node_t3)
  trial_data$valid_stop_t2 <- !is.na(trial_data$stop_t2)
  trial_data$valid_stop_t3 <- !is.na(trial_data$stop_t3)
  trial_data$continued_to_t3 <- !as_logical_col(trial_data$stop_t2)
  trial_data$continued_at_t3 <- !as_logical_col(trial_data$stop_t3)
  trial_data$better_path <- ifelse(
    trial_data$reward_t1 > trial_data$reward_t2,
    trial_data$path_t1,
    trial_data$path_t2
  )
  trial_data$continue_better_path_t3 <- as.numeric(trial_data$path_t3 == trial_data$better_path)
  trial_data <- trial_data[
    !is.na(trial_data$reward_t1) &
      !is.na(trial_data$reward_t2) &
      trial_data$reward_t1 != trial_data$reward_t2 &
      !is.na(trial_data$path_t1) &
      !is.na(trial_data$path_t2) &
      !is.na(trial_data$path_t3) &
      trial_data$path_t1 != trial_data$path_t2 &
      trial_data$valid_stop_t2 &
      trial_data$valid_stop_t3 &
      trial_data$continued_to_t3 &
      trial_data$continued_at_t3,
    ,
    drop = FALSE
  ]

  if (nrow(trial_data) == 0) {
    warning("No disjoint2x2 trials continued at t3 after observing different paths with unequal rewards; better-path heatmaps will be empty.")
    return(empty_disjoint2x2_t3_different_path_reward_summary("p_continue_better_path_t3"))
  }

  t3_summary <- aggregate(
    continue_better_path_t3 ~ beta + opportunity + reward_t1 + reward_t2,
    data = trial_data,
    FUN = mean
  )
  names(t3_summary)[names(t3_summary) == "continue_better_path_t3"] <- "p_continue_better_path_t3"

  t3_counts <- aggregate(
    continue_better_path_t3 ~ beta + opportunity + reward_t1 + reward_t2,
    data = trial_data,
    FUN = length
  )
  names(t3_counts)[names(t3_counts) == "continue_better_path_t3"] <- "n"

  merge(t3_summary, t3_counts, by = c("beta", "opportunity", "reward_t1", "reward_t2"))
}

complete_observed_disjoint_path_reward <- function(nodes, rewards) {
  nodes <- suppressWarnings(as.numeric(nodes))
  rewards <- suppressWarnings(as.numeric(rewards))
  valid <- !is.na(nodes) & !is.na(rewards)
  nodes <- nodes[valid]
  rewards <- rewards[valid]
  if (length(nodes) == 0) {
    return(NA_real_)
  }

  path_rewards <- c(NA_real_, NA_real_)
  for (path_i in 0:1) {
    path_nodes <- c(path_i * 2 + 1, path_i * 2 + 2)
    if (all(path_nodes %in% nodes)) {
      path_rewards[[path_i + 1]] <- sum(rewards[match(path_nodes, nodes)])
    }
  }

  path_rewards <- path_rewards[!is.na(path_rewards)]
  if (length(path_rewards) == 0) {
    return(NA_real_)
  }
  path_rewards[[1]]
}

build_disjoint2x2_t4_observed_path_reward_summary <- function(dat) {
  required_cols <- c(
    "expanded_node_t1",
    "expanded_reward_t1",
    "expanded_node_t2",
    "expanded_reward_t2",
    "expanded_node_t3",
    "expanded_reward_t3",
    "stop_t3",
    "stop_t4"
  )
  if (any(!required_cols %in% names(dat))) {
    warning(
      "Cannot build disjoint2x2 t4 observed-path continue plot; expected expanded_node/reward_t1/t2/t3 and stop_t3/t4 columns."
    )
    return(empty_disjoint2x2_t4_observed_path_reward_summary())
  }

  trial_id_cols <- intersect(c("beta", "opportunity", "seed", "graph"), names(dat))
  trial_cols <- unique(c(trial_id_cols, required_cols))
  trial_data <- unique(dat[, trial_cols, drop = FALSE])
  trial_data$valid_stop_t3 <- !is.na(trial_data$stop_t3)
  trial_data$valid_stop_t4 <- !is.na(trial_data$stop_t4)
  trial_data$continued_to_t4 <- !as_logical_col(trial_data$stop_t3)
  trial_data$continue_t4 <- as.numeric(!as_logical_col(trial_data$stop_t4))
  trial_data$observed_path_reward <- vapply(seq_len(nrow(trial_data)), function(row_i) {
    complete_observed_disjoint_path_reward(
      c(
        trial_data$expanded_node_t1[[row_i]],
        trial_data$expanded_node_t2[[row_i]],
        trial_data$expanded_node_t3[[row_i]]
      ),
      c(
        trial_data$expanded_reward_t1[[row_i]],
        trial_data$expanded_reward_t2[[row_i]],
        trial_data$expanded_reward_t3[[row_i]]
      )
    )
  }, numeric(1))
  trial_data <- trial_data[
    !is.na(trial_data$observed_path_reward) &
      trial_data$valid_stop_t3 &
      trial_data$valid_stop_t4 &
      trial_data$continued_to_t4,
    ,
    drop = FALSE
  ]

  if (nrow(trial_data) == 0) {
    warning("No disjoint2x2 trials reached t4 with a fully observed path; t4 observed-path continue plot will be empty.")
    return(empty_disjoint2x2_t4_observed_path_reward_summary())
  }

  t4_summary <- aggregate(
    continue_t4 ~ beta + opportunity + observed_path_reward,
    data = trial_data,
    FUN = mean
  )
  names(t4_summary)[names(t4_summary) == "continue_t4"] <- "p_continue_t4"

  t4_counts <- aggregate(
    continue_t4 ~ beta + opportunity + observed_path_reward,
    data = trial_data,
    FUN = length
  )
  names(t4_counts)[names(t4_counts) == "continue_t4"] <- "n"

  merge(t4_summary, t4_counts, by = c("beta", "opportunity", "observed_path_reward"))
}

disjoint2x2_t3_same_path_reward_summary <- if (is_disjoint2x2) {
  build_disjoint2x2_t3_same_path_reward_summary(all_data)
} else {
  empty_disjoint2x2_t3_same_path_reward_summary()
}

disjoint2x2_t3_different_path_reward_summary <- if (is_disjoint2x2) {
  build_disjoint2x2_t3_different_path_reward_summary(all_data)
} else {
  empty_disjoint2x2_t3_different_path_reward_summary("p_continue_t3")
}

disjoint2x2_t3_better_path_continue_summary <- if (is_disjoint2x2) {
  build_disjoint2x2_t3_better_path_continue_summary(all_data)
} else {
  empty_disjoint2x2_t3_different_path_reward_summary("p_continue_better_path_t3")
}

disjoint2x2_t4_observed_path_reward_summary <- if (is_disjoint2x2) {
  build_disjoint2x2_t4_observed_path_reward_summary(all_data)
} else {
  empty_disjoint2x2_t4_observed_path_reward_summary()
}

disjoint2x2_path_value_diff_continue_summary <- if (is_disjoint2x2) {
  build_disjoint2x2_path_value_diff_continue_summary(all_data)
} else {
  empty_disjoint2x2_path_value_diff_continue_summary()
}

disjoint3x2_t1_path_action_summary <- if (is_disjoint3x2) {
  build_disjoint3x2_t1_path_action_summary(all_data)
} else {
  empty_disjoint3x2_t1_path_action_summary()
}

disjoint3x2_later_path_action_summaries <- if (is_disjoint3x2) {
  build_disjoint3x2_later_path_action_summaries(all_data)
} else {
  list(
    incomplete = empty_disjoint3x2_later_path_action_summary(FALSE),
    complete = empty_disjoint3x2_later_path_action_summary(TRUE)
  )
}
disjoint3x2_later_incomplete_path_action_summary <- disjoint3x2_later_path_action_summaries$incomplete
disjoint3x2_later_complete_path_action_summary <- disjoint3x2_later_path_action_summaries$complete

disjoint3x2_best_path_continue_summary <- if (is_disjoint3x2) {
  build_disjoint3x2_best_path_continue_summary(all_data)
} else {
  empty_disjoint3x2_best_path_continue_summary()
}

disjoint_ever_second_node_summary <- if (is_disjoint_tree) {
  build_disjoint_ever_second_node_summary(all_data)
} else {
  empty_disjoint_ever_second_node_summary()
}

legacy6_continue_node_summary <- if (is_legacy6_tree) {
  build_legacy6_continue_node_summary(all_data)
} else {
  empty_legacy6_continue_node_summary()
}

legacy6_path_action_by_reward_summary <- if (is_legacy6_tree) {
  build_legacy6_path_action_by_reward_summary(all_data)
} else {
  empty_legacy6_path_action_by_reward_summary()
}

legacy6_value_continue_summaries <- if (is_legacy6_tree) {
  build_legacy6_value_continue_summaries(all_data)
} else {
  list(
    complete = empty_legacy6_value_continue_summary(),
    best_observed = empty_legacy6_value_continue_summary()
  )
}
legacy6_complete_path_continue_summary <- legacy6_value_continue_summaries$complete
legacy6_best_observed_path_continue_summary <- legacy6_value_continue_summaries$best_observed

build_t3_max_conditioned_continue_summary <- function(dat) {
  required_cols <- c("expanded_reward_t1", "expanded_reward_t2", "stop_t3")
  if (any(!required_cols %in% names(dat))) {
    warning(
      "Cannot build t3 max-reward conditioned continue plot; expected expanded_reward_t1, expanded_reward_t2, and stop_t3 columns."
    )
    return(data.frame(
      beta = character(),
      opportunity = character(),
      max_reward_observed = numeric(),
      p_continue_t3 = numeric(),
      n = integer(),
      stringsAsFactors = FALSE
    ))
  }

  trial_id_cols <- intersect(c("beta", "opportunity", "seed", "graph"), names(dat))
  trial_cols <- unique(c(trial_id_cols, required_cols))
  trial_data <- unique(dat[, trial_cols, drop = FALSE])
  trial_data$reward_t1 <- suppressWarnings(as.numeric(trial_data$expanded_reward_t1))
  trial_data$reward_t2 <- suppressWarnings(as.numeric(trial_data$expanded_reward_t2))
  stop_t3_raw <- trial_data$stop_t3
  trial_data$valid_stop_t3 <- !is.na(stop_t3_raw)
  trial_data$continue_t3 <- as.numeric(!as_logical_col(stop_t3_raw))
  trial_data$max_reward_observed <- pmax(
    trial_data$reward_t1,
    trial_data$reward_t2,
    na.rm = TRUE
  )
  trial_data <- trial_data[
    !is.na(trial_data$reward_t1) &
      !is.na(trial_data$reward_t2) &
      trial_data$valid_stop_t3,
    ,
    drop = FALSE
  ]

  if (nrow(trial_data) == 0) {
    warning("No trials had rewards t1-t2 before a valid t3 decision; t3 max-reward continue plot will be empty.")
    return(data.frame(
      beta = character(),
      opportunity = character(),
      max_reward_observed = numeric(),
      p_continue_t3 = numeric(),
      n = integer(),
      stringsAsFactors = FALSE
    ))
  }

  t3_summary <- aggregate(
    continue_t3 ~ beta + opportunity + max_reward_observed,
    data = trial_data,
    FUN = mean
  )
  names(t3_summary)[names(t3_summary) == "continue_t3"] <- "p_continue_t3"

  t3_counts <- aggregate(
    continue_t3 ~ beta + opportunity + max_reward_observed,
    data = trial_data,
    FUN = length
  )
  names(t3_counts)[names(t3_counts) == "continue_t3"] <- "n"

  merge(
    t3_summary,
    t3_counts,
    by = c("beta", "opportunity", "max_reward_observed")
  )
}

continue_t3_max_conditioned_summary <- if (is_bandit3) {
  build_t3_max_conditioned_continue_summary(all_data)
} else {
  data.frame(
    beta = character(),
    opportunity = character(),
    max_reward_observed = numeric(),
    p_continue_t3 = numeric(),
    n = integer(),
    stringsAsFactors = FALSE
  )
}

build_t4_max_conditioned_continue_summary <- function(dat) {
  required_cols <- c(
    "expanded_reward_t1",
    "expanded_reward_t2",
    "expanded_reward_t3",
    "stop_t4"
  )
  if (any(!required_cols %in% names(dat))) {
    warning(
      "Cannot build t4 max-reward conditioned continue plot; expected expanded_reward_t1, expanded_reward_t2, expanded_reward_t3, and stop_t4 columns."
    )
    return(data.frame(
      beta = character(),
      opportunity = character(),
      max_reward_observed = numeric(),
      p_continue_t4 = numeric(),
      n = integer(),
      stringsAsFactors = FALSE
    ))
  }

  trial_id_cols <- intersect(c("beta", "opportunity", "seed", "graph"), names(dat))
  trial_cols <- unique(c(trial_id_cols, required_cols))
  trial_data <- unique(dat[, trial_cols, drop = FALSE])
  trial_data$reward_t1 <- suppressWarnings(as.numeric(trial_data$expanded_reward_t1))
  trial_data$reward_t2 <- suppressWarnings(as.numeric(trial_data$expanded_reward_t2))
  trial_data$reward_t3 <- suppressWarnings(as.numeric(trial_data$expanded_reward_t3))
  stop_t4_raw <- trial_data$stop_t4
  trial_data$valid_stop_t4 <- !is.na(stop_t4_raw)
  trial_data$continue_t4 <- as.numeric(!as_logical_col(stop_t4_raw))
  trial_data$max_reward_observed <- pmax(
    trial_data$reward_t1,
    trial_data$reward_t2,
    trial_data$reward_t3,
    na.rm = TRUE
  )
  trial_data <- trial_data[
    !is.na(trial_data$reward_t1) &
      !is.na(trial_data$reward_t2) &
      !is.na(trial_data$reward_t3) &
      trial_data$valid_stop_t4,
    ,
    drop = FALSE
  ]

  if (nrow(trial_data) == 0) {
    warning("No trials had rewards t1-t3 before a valid t4 decision; t4 max-reward continue plot will be empty.")
    return(data.frame(
      beta = character(),
      opportunity = character(),
      max_reward_observed = numeric(),
      p_continue_t4 = numeric(),
      n = integer(),
      stringsAsFactors = FALSE
    ))
  }

  t4_summary <- aggregate(
    continue_t4 ~ beta + opportunity + max_reward_observed,
    data = trial_data,
    FUN = mean
  )
  names(t4_summary)[names(t4_summary) == "continue_t4"] <- "p_continue_t4"

  t4_counts <- aggregate(
    continue_t4 ~ beta + opportunity + max_reward_observed,
    data = trial_data,
    FUN = length
  )
  names(t4_counts)[names(t4_counts) == "continue_t4"] <- "n"

  merge(
    t4_summary,
    t4_counts,
    by = c("beta", "opportunity", "max_reward_observed")
  )
}

continue_t4_max_conditioned_summary <- if (is_bandit4) {
  build_t4_max_conditioned_continue_summary(all_data)
} else {
  data.frame(
    beta = character(),
    opportunity = character(),
    max_reward_observed = numeric(),
    p_continue_t4 = numeric(),
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
  kl_cols <- kl_columns(dat)

  if (length(kl_cols) == 0) {
    stop(
      paste(
        "Expected kl_d_obs_t* or kl_d_t* columns for KL plotting, but none were found.",
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
      timestep <- kl_timestep(kl_col)
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

kl_columns <- function(dat) {
  obs_cols <- grep("^kl_d_obs_t[0-9]+$", names(dat), value = TRUE)
  if (length(obs_cols) > 0) {
    return(obs_cols[order(as.integer(sub("^kl_d_obs_t", "", obs_cols)))])
  }
  paid_cols <- grep("^kl_d_t[0-9]+$", names(dat), value = TRUE)
  paid_cols[order(as.integer(sub("^kl_d_t", "", paid_cols)))]
}

kl_column_for_timestep <- function(dat, timestep) {
  obs_col <- sprintf("kl_d_obs_t%d", timestep)
  if (obs_col %in% names(dat)) {
    return(obs_col)
  }
  paid_col <- sprintf("kl_d_t%d", timestep)
  if (paid_col %in% names(dat)) {
    return(paid_col)
  }
  NA_character_
}

kl_timestep <- function(kl_col) {
  as.integer(sub("^kl_d(_obs)?_t", "", kl_col))
}

kl_summary <- build_kl_summary(all_data)

build_kl_by_reward_summary <- function(dat, timestep = 1) {
  reward_col <- sprintf("expanded_reward_t%d", timestep)
  kl_col <- kl_column_for_timestep(dat, timestep)

  if (is.na(kl_col)) {
    stop(sprintf("Cannot compute KL-by-reward summary. Missing KL column for timestep %d.", timestep))
  }
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

build_kl_by_reward_all_summary <- function(dat) {
  reward_cols <- grep("^expanded_reward_t[0-9]+$", names(dat), value = TRUE)
  reward_cols <- reward_cols[order(as.integer(sub("^expanded_reward_t", "", reward_cols)))]
  kl_cols <- kl_columns(dat)
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
      timestep <- kl_timestep(kl_cols[[t]])
      reward_col <- sprintf("expanded_reward_t%d", timestep)
      if (!reward_col %in% names(trial_data)) {
        next
      }
      reward_value <- suppressWarnings(as.numeric(trial_data[[reward_col]][[i]]))
      kl_value <- suppressWarnings(as.numeric(trial_data[[kl_cols[[t]]]][[i]]))
      if (!is.na(reward_value) && !is.na(kl_value)) {
        rows[[row_i]] <- data.frame(
          beta = trial_data$beta[[i]],
          opportunity = trial_data$opportunity[[i]],
          timestep = timestep,
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

build_deep_probe_accuracy_summary <- function(dat) {
  probe_sources <- c("lstm", "decoder")
  rows <- list()
  row_i <- 1

  eval_data <- dat
  if ("deep_probe_split" %in% names(eval_data)) {
    eval_data <- eval_data[
      is.na(eval_data$deep_probe_split) |
        eval_data$deep_probe_split == "test",
      ,
      drop = FALSE
    ]
  }

  for (source_name in probe_sources) {
    correct_cols <- grep(
      sprintf("^%s_deep_probe_correct_t[0-9]+$", source_name),
      names(eval_data),
      value = TRUE
    )
    correct_cols <- correct_cols[order(as.integer(sub(sprintf("^%s_deep_probe_correct_t", source_name), "", correct_cols)))]

    if (length(correct_cols) == 0 || !"actual_reward" %in% names(eval_data)) {
      next
    }

    expanded_node_cols <- grep("^expanded_node_t[0-9]+$", names(eval_data), value = TRUE)
    expanded_node_cols <- expanded_node_cols[order(as.integer(sub("^expanded_node_t", "", expanded_node_cols)))]
    id_cols <- intersect(c("beta", "opportunity", "seed", "graph", "node", "actual_reward"), names(eval_data))
    node_data <- unique(eval_data[, unique(c(id_cols, expanded_node_cols, correct_cols)), drop = FALSE])

    for (i in seq_len(nrow(node_data))) {
      actual_reward <- suppressWarnings(as.numeric(node_data$actual_reward[[i]]))
      node_value <- suppressWarnings(as.numeric(node_data$node[[i]]))
      if (is.na(actual_reward) || is.na(node_value)) {
        next
      }

      observed_timestep <- NA_integer_
      for (expanded_node_col in expanded_node_cols) {
        expanded_node <- suppressWarnings(as.numeric(node_data[[expanded_node_col]][[i]]))
        if (!is.na(expanded_node) && expanded_node == node_value) {
          observed_timestep <- as.integer(sub("^expanded_node_t", "", expanded_node_col))
          break
        }
      }
      if (is.na(observed_timestep)) {
        next
      }

      for (correct_col in correct_cols) {
        timestep <- as.integer(sub(sprintf("^%s_deep_probe_correct_t", source_name), "", correct_col))
        correct <- suppressWarnings(as.numeric(node_data[[correct_col]][[i]]))
        if (!is.na(correct)) {
          rows[[row_i]] <- data.frame(
            beta = node_data$beta[[i]],
            opportunity = node_data$opportunity[[i]],
            source = source_name,
            observed_timestep = observed_timestep,
            timestep = timestep,
            reward = actual_reward,
            correct = correct,
            stringsAsFactors = FALSE
          )
          row_i <- row_i + 1
        }
      }
    }
  }

  if (length(rows) == 0) {
    warning("No deep probe columns were found; deep probe accuracy panels will be empty.")
    return(data.frame(
      beta = character(),
      opportunity = character(),
      source = character(),
      observed_timestep = integer(),
      timestep = integer(),
      reward = numeric(),
      accuracy = numeric(),
      n = integer(),
      stringsAsFactors = FALSE
    ))
  }

  probe_data <- do.call(rbind, rows)
  probe_summary <- aggregate(
    correct ~ beta + opportunity + source + observed_timestep + timestep + reward,
    data = probe_data,
    FUN = mean
  )
  names(probe_summary)[names(probe_summary) == "correct"] <- "accuracy"
  probe_counts <- aggregate(
    correct ~ beta + opportunity + source + observed_timestep + timestep + reward,
    data = probe_data,
    FUN = length
  )
  names(probe_counts)[names(probe_counts) == "correct"] <- "n"
  merge(
    probe_summary,
    probe_counts,
    by = c("beta", "opportunity", "source", "observed_timestep", "timestep", "reward")
  )
}

build_deep_probe_confusion_summary <- function(dat) {
  probe_sources <- c("lstm", "decoder")
  rows <- list()
  row_i <- 1

  eval_data <- dat
  if ("deep_probe_split" %in% names(eval_data)) {
    eval_data <- eval_data[
      is.na(eval_data$deep_probe_split) |
        eval_data$deep_probe_split == "test",
      ,
      drop = FALSE
    ]
  }

  for (source_name in probe_sources) {
    pred_cols <- grep(
      sprintf("^%s_deep_probe_pred_reward_t[0-9]+$", source_name),
      names(eval_data),
      value = TRUE
    )
    pred_cols <- pred_cols[order(as.integer(sub(sprintf("^%s_deep_probe_pred_reward_t", source_name), "", pred_cols)))]

    if (length(pred_cols) == 0 || !"actual_reward" %in% names(eval_data)) {
      next
    }

    expanded_node_cols <- grep("^expanded_node_t[0-9]+$", names(eval_data), value = TRUE)
    expanded_node_cols <- expanded_node_cols[order(as.integer(sub("^expanded_node_t", "", expanded_node_cols)))]
    id_cols <- intersect(c("beta", "opportunity", "seed", "graph", "node", "actual_reward"), names(eval_data))
    node_data <- unique(eval_data[, unique(c(id_cols, expanded_node_cols, pred_cols)), drop = FALSE])

    for (i in seq_len(nrow(node_data))) {
      true_reward <- suppressWarnings(as.numeric(node_data$actual_reward[[i]]))
      node_value <- suppressWarnings(as.numeric(node_data$node[[i]]))
      if (is.na(true_reward) || is.na(node_value)) {
        next
      }

      observed_timestep <- NA_integer_
      for (expanded_node_col in expanded_node_cols) {
        expanded_node <- suppressWarnings(as.numeric(node_data[[expanded_node_col]][[i]]))
        if (!is.na(expanded_node) && expanded_node == node_value) {
          observed_timestep <- as.integer(sub("^expanded_node_t", "", expanded_node_col))
          break
        }
      }
      if (is.na(observed_timestep)) {
        next
      }

      for (pred_col in pred_cols) {
        timestep <- as.integer(sub(sprintf("^%s_deep_probe_pred_reward_t", source_name), "", pred_col))
        pred_reward <- suppressWarnings(as.numeric(node_data[[pred_col]][[i]]))
        if (!is.na(pred_reward) && timestep >= observed_timestep) {
          rows[[row_i]] <- data.frame(
            beta = node_data$beta[[i]],
            opportunity = node_data$opportunity[[i]],
            source = source_name,
            observed_timestep = observed_timestep,
            timestep = timestep,
            true_reward = true_reward,
            pred_reward = pred_reward,
            count = 1L,
            stringsAsFactors = FALSE
          )
          row_i <- row_i + 1
        }
      }
    }
  }

  if (length(rows) == 0) {
    warning("No deep probe prediction columns were found; deep probe confusion matrices will be empty.")
    return(data.frame(
      beta = character(),
      opportunity = character(),
      source = character(),
      observed_timestep = integer(),
      timestep = integer(),
      true_reward = numeric(),
      pred_reward = numeric(),
      n = integer(),
      total_n = integer(),
      probability = numeric(),
      stringsAsFactors = FALSE
    ))
  }

  probe_data <- do.call(rbind, rows)
  confusion_counts <- aggregate(
    count ~ beta + opportunity + source + observed_timestep + timestep + true_reward + pred_reward,
    data = probe_data,
    FUN = sum
  )
  names(confusion_counts)[names(confusion_counts) == "count"] <- "n"
  true_counts <- aggregate(
    n ~ beta + opportunity + source + observed_timestep + timestep + true_reward,
    data = confusion_counts,
    FUN = sum
  )
  names(true_counts)[names(true_counts) == "n"] <- "total_n"
  confusion_summary <- merge(
    confusion_counts,
    true_counts,
    by = c("beta", "opportunity", "source", "observed_timestep", "timestep", "true_reward")
  )
  confusion_summary$probability <- ifelse(
    confusion_summary$total_n > 0,
    confusion_summary$n / confusion_summary$total_n,
    NA_real_
  )
  confusion_summary
}

build_deep_probe_signed_error_summary <- function(dat) {
  probe_sources <- c("lstm", "decoder")
  rows <- list()
  row_i <- 1

  eval_data <- dat
  if ("deep_probe_split" %in% names(eval_data)) {
    eval_data <- eval_data[
      is.na(eval_data$deep_probe_split) |
        eval_data$deep_probe_split == "test",
      ,
      drop = FALSE
    ]
  }

  for (source_name in probe_sources) {
    pred_cols <- grep(
      sprintf("^%s_deep_probe_pred_reward_t[0-9]+$", source_name),
      names(eval_data),
      value = TRUE
    )
    pred_cols <- pred_cols[order(as.integer(sub(sprintf("^%s_deep_probe_pred_reward_t", source_name), "", pred_cols)))]

    if (length(pred_cols) == 0 || !"actual_reward" %in% names(eval_data)) {
      next
    }

    expanded_node_cols <- grep("^expanded_node_t[0-9]+$", names(eval_data), value = TRUE)
    expanded_node_cols <- expanded_node_cols[order(as.integer(sub("^expanded_node_t", "", expanded_node_cols)))]
    id_cols <- intersect(c("beta", "opportunity", "seed", "graph", "node", "actual_reward"), names(eval_data))
    node_data <- unique(eval_data[, unique(c(id_cols, expanded_node_cols, pred_cols)), drop = FALSE])

    for (i in seq_len(nrow(node_data))) {
      true_reward <- suppressWarnings(as.numeric(node_data$actual_reward[[i]]))
      node_value <- suppressWarnings(as.numeric(node_data$node[[i]]))
      if (is.na(true_reward) || is.na(node_value)) {
        next
      }

      observed_timestep <- NA_integer_
      for (expanded_node_col in expanded_node_cols) {
        expanded_node <- suppressWarnings(as.numeric(node_data[[expanded_node_col]][[i]]))
        if (!is.na(expanded_node) && expanded_node == node_value) {
          observed_timestep <- as.integer(sub("^expanded_node_t", "", expanded_node_col))
          break
        }
      }
      if (is.na(observed_timestep)) {
        next
      }

      for (pred_col in pred_cols) {
        timestep <- as.integer(sub(sprintf("^%s_deep_probe_pred_reward_t", source_name), "", pred_col))
        pred_reward <- suppressWarnings(as.numeric(node_data[[pred_col]][[i]]))
        if (!is.na(pred_reward) && timestep >= observed_timestep) {
          rows[[row_i]] <- data.frame(
            beta = node_data$beta[[i]],
            opportunity = node_data$opportunity[[i]],
            source = source_name,
            observed_timestep = observed_timestep,
            timestep = timestep,
            reward = true_reward,
            signed_error = pred_reward - true_reward,
            stringsAsFactors = FALSE
          )
          row_i <- row_i + 1
        }
      }
    }
  }

  if (length(rows) == 0) {
    warning("No deep probe prediction columns were found; deep probe signed-error panels will be empty.")
    return(data.frame(
      beta = character(),
      opportunity = character(),
      source = character(),
      observed_timestep = integer(),
      timestep = integer(),
      reward = numeric(),
      mean_signed_error = numeric(),
      sd_signed_error = numeric(),
      n = integer(),
      stringsAsFactors = FALSE
    ))
  }

  probe_data <- do.call(rbind, rows)
  error_summary <- aggregate(
    signed_error ~ beta + opportunity + source + observed_timestep + timestep + reward,
    data = probe_data,
    FUN = mean
  )
  names(error_summary)[names(error_summary) == "signed_error"] <- "mean_signed_error"
  error_sd <- aggregate(
    signed_error ~ beta + opportunity + source + observed_timestep + timestep + reward,
    data = probe_data,
    FUN = sd
  )
  names(error_sd)[names(error_sd) == "signed_error"] <- "sd_signed_error"
  error_counts <- aggregate(
    signed_error ~ beta + opportunity + source + observed_timestep + timestep + reward,
    data = probe_data,
    FUN = length
  )
  names(error_counts)[names(error_counts) == "signed_error"] <- "n"

  error_summary <- merge(
    error_summary,
    error_sd,
    by = c("beta", "opportunity", "source", "observed_timestep", "timestep", "reward")
  )
  merge(
    error_summary,
    error_counts,
    by = c("beta", "opportunity", "source", "observed_timestep", "timestep", "reward")
  )
}

empty_deep_probe_path_context_signed_error_summary <- function() {
  data.frame(
    beta = character(),
    opportunity = character(),
    source = character(),
    context_type = character(),
    observed_timestep = integer(),
    timestep = integer(),
    context_reward = numeric(),
    mean_signed_error = numeric(),
    sd_signed_error = numeric(),
    n = integer(),
    stringsAsFactors = FALSE
  )
}

build_deep_probe_path_context_signed_error_summary <- function(dat) {
  if (!is_disjoint2x2) {
    return(empty_deep_probe_path_context_signed_error_summary())
  }

  probe_sources <- c("lstm", "decoder")
  expanded_node_cols <- grep("^expanded_node_t[0-9]+$", names(dat), value = TRUE)
  expanded_node_cols <- expanded_node_cols[
    order(as.integer(sub("^expanded_node_t", "", expanded_node_cols)))
  ]

  if (
    length(expanded_node_cols) == 0 ||
      any(!c("node", "actual_reward") %in% names(dat))
  ) {
    warning(
      "Cannot build disjoint2x2 path-context signed-error panels; expected node, actual_reward, and expanded_node_t* columns."
    )
    return(empty_deep_probe_path_context_signed_error_summary())
  }

  eval_data <- dat
  if ("deep_probe_split" %in% names(eval_data)) {
    eval_data <- eval_data[
      is.na(eval_data$deep_probe_split) |
        eval_data$deep_probe_split == "test",
      ,
      drop = FALSE
    ]
  }

  reward_for_node <- function(node_rewards, node) {
    value <- node_rewards[as.character(node)]
    if (length(value) == 0 || is.na(value[[1]])) {
      return(NA_real_)
    }
    unname(value[[1]])
  }

  rows <- list()
  row_i <- 1
  trial_id_cols <- intersect(c("beta", "opportunity", "seed", "graph"), names(eval_data))

  for (source_name in probe_sources) {
    pred_cols <- grep(
      sprintf("^%s_deep_probe_pred_reward_t[0-9]+$", source_name),
      names(eval_data),
      value = TRUE
    )
    pred_cols <- pred_cols[
      order(as.integer(sub(sprintf("^%s_deep_probe_pred_reward_t", source_name), "", pred_cols)))
    ]
    if (length(pred_cols) == 0) {
      next
    }

    id_cols <- unique(c(trial_id_cols, "node", "actual_reward"))
    node_data <- unique(eval_data[, unique(c(id_cols, expanded_node_cols, pred_cols)), drop = FALSE])
    if (nrow(node_data) == 0) {
      next
    }

    if (length(trial_id_cols) > 0) {
      node_data$trial_key <- do.call(
        paste,
        c(node_data[, trial_id_cols, drop = FALSE], sep = "\r")
      )
    } else {
      node_data$trial_key <- seq_len(nrow(node_data))
    }

    trial_keys <- unique(node_data$trial_key)
    for (trial_key in trial_keys) {
      trial_rows <- node_data[node_data$trial_key == trial_key, , drop = FALSE]
      if (nrow(trial_rows) == 0) {
        next
      }

      trial_nodes <- suppressWarnings(as.numeric(trial_rows$node))
      trial_rewards <- suppressWarnings(as.numeric(trial_rows$actual_reward))
      valid_trial_nodes <- !is.na(trial_nodes) & !is.na(trial_rewards)
      if (!any(valid_trial_nodes)) {
        next
      }
      node_rewards <- trial_rewards[valid_trial_nodes]
      names(node_rewards) <- as.character(trial_nodes[valid_trial_nodes])

      expanded_nodes <- suppressWarnings(as.numeric(unlist(
        trial_rows[1, expanded_node_cols, drop = FALSE],
        use.names = FALSE
      )))
      names(expanded_nodes) <- as.integer(sub("^expanded_node_t", "", expanded_node_cols))
      expanded_nodes <- expanded_nodes[order(as.integer(names(expanded_nodes)))]

      for (i in seq_len(nrow(trial_rows))) {
        target_node <- suppressWarnings(as.numeric(trial_rows$node[[i]]))
        true_reward <- suppressWarnings(as.numeric(trial_rows$actual_reward[[i]]))
        target_path <- disjoint_path_id(target_node)
        if (is.na(target_node) || is.na(true_reward) || is.na(target_path)) {
          next
        }

        observed_timestep <- NA_integer_
        for (expanded_node_col in expanded_node_cols) {
          expanded_node <- suppressWarnings(as.numeric(trial_rows[[expanded_node_col]][[i]]))
          if (!is.na(expanded_node) && expanded_node == target_node) {
            observed_timestep <- as.integer(sub("^expanded_node_t", "", expanded_node_col))
            break
          }
        }
        if (is.na(observed_timestep)) {
          next
        }

        target_path_nodes <- as.numeric(target_path) * 2 + c(1, 2)
        same_path_other_node <- target_path_nodes[target_path_nodes != target_node]
        other_path_nodes <- (1 - as.numeric(target_path)) * 2 + c(1, 2)

        for (pred_col in pred_cols) {
          timestep <- as.integer(sub(sprintf("^%s_deep_probe_pred_reward_t", source_name), "", pred_col))
          pred_reward <- suppressWarnings(as.numeric(trial_rows[[pred_col]][[i]]))
          if (is.na(pred_reward) || is.na(timestep) || timestep < observed_timestep) {
            next
          }

          observed_nodes <- expanded_nodes[as.integer(names(expanded_nodes)) <= timestep]
          observed_nodes <- observed_nodes[!is.na(observed_nodes)]
          signed_error <- pred_reward - true_reward

          add_context_row <- function(context_type, context_reward) {
            if (is.na(context_reward)) {
              return(NULL)
            }
            rows[[row_i]] <<- data.frame(
              beta = trial_rows$beta[[i]],
              opportunity = trial_rows$opportunity[[i]],
              source = source_name,
              context_type = context_type,
              observed_timestep = observed_timestep,
              timestep = timestep,
              context_reward = context_reward,
              signed_error = signed_error,
              stringsAsFactors = FALSE
            )
            row_i <<- row_i + 1
            NULL
          }

          if (same_path_other_node %in% observed_nodes) {
            add_context_row(
              "same_path_other_observed",
              reward_for_node(node_rewards, same_path_other_node)
            )
          }

          observed_other_nodes <- other_path_nodes[other_path_nodes %in% observed_nodes]
          if (length(observed_other_nodes) == 1) {
            add_context_row(
              "other_path_one_observed",
              reward_for_node(node_rewards, observed_other_nodes[[1]])
            )
          } else if (length(observed_other_nodes) == 2) {
            other_path_rewards <- vapply(
              observed_other_nodes,
              function(node) reward_for_node(node_rewards, node),
              numeric(1)
            )
            if (all(!is.na(other_path_rewards))) {
              add_context_row("other_path_both_observed", sum(other_path_rewards))
            }
          }
        }
      }
    }
  }

  if (length(rows) == 0) {
    warning("No disjoint2x2 path-context signed-error values were found.")
    return(empty_deep_probe_path_context_signed_error_summary())
  }

  probe_data <- do.call(rbind, rows)
  group_cols <- c(
    "beta",
    "opportunity",
    "source",
    "context_type",
    "observed_timestep",
    "timestep",
    "context_reward"
  )

  error_summary <- aggregate(
    signed_error ~ beta + opportunity + source + context_type + observed_timestep + timestep + context_reward,
    data = probe_data,
    FUN = mean
  )
  names(error_summary)[names(error_summary) == "signed_error"] <- "mean_signed_error"
  error_sd <- aggregate(
    signed_error ~ beta + opportunity + source + context_type + observed_timestep + timestep + context_reward,
    data = probe_data,
    FUN = sd
  )
  names(error_sd)[names(error_sd) == "signed_error"] <- "sd_signed_error"
  error_counts <- aggregate(
    signed_error ~ beta + opportunity + source + context_type + observed_timestep + timestep + context_reward,
    data = probe_data,
    FUN = length
  )
  names(error_counts)[names(error_counts) == "signed_error"] <- "n"

  error_summary <- merge(error_summary, error_sd, by = group_cols)
  merge(error_summary, error_counts, by = group_cols)
}

build_deep_probe_t2_conditioned_summary <- function(dat) {
  probe_sources <- c("lstm", "decoder")
  required_cols <- c("expanded_node_t2", "expanded_reward_t1", "expanded_reward_t2", "node")
  if (any(!required_cols %in% names(dat))) {
    warning("Cannot build t2-conditioned deep probe heatmaps; expected expanded_node_t2, expanded_reward_t1, expanded_reward_t2, and node columns.")
    return(data.frame(
      beta = character(),
      opportunity = character(),
      source = character(),
      timestep = integer(),
      reward_t1 = numeric(),
      reward_t2 = numeric(),
      accuracy = numeric(),
      n = integer(),
      stringsAsFactors = FALSE
    ))
  }

  eval_data <- dat
  if ("deep_probe_split" %in% names(eval_data)) {
    eval_data <- eval_data[
      is.na(eval_data$deep_probe_split) |
        eval_data$deep_probe_split == "test",
      ,
      drop = FALSE
    ]
  }

  rows <- list()
  row_i <- 1
  for (source_name in probe_sources) {
    correct_cols <- grep(
      sprintf("^%s_deep_probe_correct_t[0-9]+$", source_name),
      names(eval_data),
      value = TRUE
    )
    correct_cols <- correct_cols[order(as.integer(sub(sprintf("^%s_deep_probe_correct_t", source_name), "", correct_cols)))]
    if (length(correct_cols) == 0) {
      next
    }

    id_cols <- intersect(c("beta", "opportunity", "seed", "graph", "node"), names(eval_data))
    node_data <- unique(eval_data[, unique(c(
      id_cols,
      "expanded_node_t2",
      "expanded_reward_t1",
      "expanded_reward_t2",
      correct_cols
    )), drop = FALSE])

    for (i in seq_len(nrow(node_data))) {
      node_value <- suppressWarnings(as.numeric(node_data$node[[i]]))
      expanded_node_t2 <- suppressWarnings(as.numeric(node_data$expanded_node_t2[[i]]))
      reward_t1 <- suppressWarnings(as.numeric(node_data$expanded_reward_t1[[i]]))
      reward_t2 <- suppressWarnings(as.numeric(node_data$expanded_reward_t2[[i]]))
      if (
        is.na(node_value) ||
          is.na(expanded_node_t2) ||
          is.na(reward_t1) ||
          is.na(reward_t2) ||
          node_value != expanded_node_t2
      ) {
        next
      }

      for (correct_col in correct_cols) {
        timestep <- as.integer(sub(sprintf("^%s_deep_probe_correct_t", source_name), "", correct_col))
        correct <- suppressWarnings(as.numeric(node_data[[correct_col]][[i]]))
        if (!is.na(correct)) {
          rows[[row_i]] <- data.frame(
            beta = node_data$beta[[i]],
            opportunity = node_data$opportunity[[i]],
            source = source_name,
            timestep = timestep,
            reward_t1 = reward_t1,
            reward_t2 = reward_t2,
            correct = correct,
            stringsAsFactors = FALSE
          )
          row_i <- row_i + 1
        }
      }
    }
  }

  if (length(rows) == 0) {
    warning("No t2-conditioned deep probe values were found; t2-conditioned heatmaps will be empty.")
    return(data.frame(
      beta = character(),
      opportunity = character(),
      source = character(),
      timestep = integer(),
      reward_t1 = numeric(),
      reward_t2 = numeric(),
      accuracy = numeric(),
      n = integer(),
      stringsAsFactors = FALSE
    ))
  }

  probe_data <- do.call(rbind, rows)
  probe_summary <- aggregate(
    correct ~ beta + opportunity + source + timestep + reward_t1 + reward_t2,
    data = probe_data,
    FUN = mean
  )
  names(probe_summary)[names(probe_summary) == "correct"] <- "accuracy"
  probe_counts <- aggregate(
    correct ~ beta + opportunity + source + timestep + reward_t1 + reward_t2,
    data = probe_data,
    FUN = length
  )
  names(probe_counts)[names(probe_counts) == "correct"] <- "n"
  merge(
    probe_summary,
    probe_counts,
    by = c("beta", "opportunity", "source", "timestep", "reward_t1", "reward_t2")
  )
}

build_kl_transition_heatmap_summary <- function(dat) {
  timesteps <- kl_transition_timesteps(dat)
  if (length(timesteps) == 0) {
    warning("Cannot build KL transition heatmaps; no adjacent expanded_reward_t*/KL timestep pairs were found.")
    return(data.frame(
      beta = character(),
      opportunity = character(),
      timestep = integer(),
      previous_timestep = integer(),
      reward_previous = numeric(),
      reward_current = numeric(),
      kl_d = numeric(),
      n = integer(),
      stringsAsFactors = FALSE
    ))
  }

  trial_id_cols <- intersect(c("beta", "opportunity", "seed", "graph"), names(dat))
  rows <- list()
  row_i <- 1
  for (timestep in timesteps) {
    previous_timestep <- timestep - 1
    previous_reward_col <- sprintf("expanded_reward_t%d", previous_timestep)
    current_reward_col <- sprintf("expanded_reward_t%d", timestep)
    kl_col <- kl_column_for_timestep(dat, timestep)
    required_cols <- c(previous_reward_col, current_reward_col, kl_col)
    if (is.na(kl_col) || any(!required_cols %in% names(dat))) {
      next
    }

    trial_cols <- unique(c(trial_id_cols, previous_reward_col, current_reward_col, kl_col))
    trial_data <- unique(dat[, trial_cols, drop = FALSE])
    for (i in seq_len(nrow(trial_data))) {
      reward_previous <- suppressWarnings(as.numeric(trial_data[[previous_reward_col]][[i]]))
      reward_current <- suppressWarnings(as.numeric(trial_data[[current_reward_col]][[i]]))
      kl_d <- suppressWarnings(as.numeric(trial_data[[kl_col]][[i]]))
      if (!is.na(reward_previous) && !is.na(reward_current) && !is.na(kl_d)) {
        rows[[row_i]] <- data.frame(
          beta = trial_data$beta[[i]],
          opportunity = trial_data$opportunity[[i]],
          timestep = timestep,
          previous_timestep = previous_timestep,
          reward_previous = reward_previous,
          reward_current = reward_current,
          kl_d = kl_d,
          stringsAsFactors = FALSE
        )
        row_i <- row_i + 1
      }
    }
  }

  if (length(rows) == 0) {
    warning("No trials with adjacent observed rewards were found for KL transition heatmaps.")
    return(data.frame(
      beta = character(),
      opportunity = character(),
      timestep = integer(),
      previous_timestep = integer(),
      reward_previous = numeric(),
      reward_current = numeric(),
      kl_d = numeric(),
      n = integer(),
      stringsAsFactors = FALSE
    ))
  }

  heat_data <- do.call(rbind, rows)
  heat_summary <- aggregate(
    kl_d ~ beta + opportunity + timestep + previous_timestep + reward_previous + reward_current,
    data = heat_data,
    FUN = mean
  )
  heat_counts <- aggregate(
    kl_d ~ beta + opportunity + timestep + previous_timestep + reward_previous + reward_current,
    data = heat_data,
    FUN = length
  )
  names(heat_counts)[names(heat_counts) == "kl_d"] <- "n"
  merge(
    heat_summary,
    heat_counts,
    by = c("beta", "opportunity", "timestep", "previous_timestep", "reward_previous", "reward_current")
  )
}

get_trial_stop_data <- function(dat) {
  stop_cols <- grep("^stop_t[0-9]+$", names(dat), value = TRUE)
  stop_cols <- stop_cols[order(as.integer(sub("^stop_t", "", stop_cols)))]
  trial_id_cols <- intersect(c("beta", "opportunity", "seed", "graph", "chosen_path"), names(dat))
  trial_data <- unique(dat[, unique(c(trial_id_cols, stop_cols)), drop = FALSE])

  first_stop <- rep(NA_integer_, nrow(trial_data))
  stop_timesteps <- as.integer(sub("^stop_t", "", stop_cols))
  for (i in seq_len(nrow(trial_data))) {
    stop_vec <- as_logical_col(unlist(trial_data[i, stop_cols, drop = TRUE]))
    stop_at <- which(stop_vec)
    if (length(stop_at) > 0) {
      first_stop[[i]] <- stop_timesteps[[stop_at[[1]]]]
    }
  }
  trial_data$stop_timestep <- first_stop
  trial_data
}

node_in_chosen_path <- function(node, chosen_path) {
  node <- suppressWarnings(as.numeric(node))
  chosen_path <- suppressWarnings(as.numeric(chosen_path))
  out <- rep(NA, length(node))
  valid <- !is.na(node) & !is.na(chosen_path)
  if (!any(valid)) {
    return(out)
  }

  node_zero_based <- node - 1
  if (tree_size == 2 || tree_config %in% c("bandit3", "bandit4")) {
    out[valid] <- node_zero_based[valid] == chosen_path[valid]
  } else if (tree_config %in% c("disjoint2x2", "disjoint3x2")) {
    out[valid] <- floor(node_zero_based[valid] / 2) == chosen_path[valid]
  } else {
    out[valid] <- node[valid] == chosen_path[valid]
  }
  out
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
  trial_stop <- trial_stop[!is.na(trial_stop$stop_timestep), , drop = FALSE]
  if (nrow(trial_stop) == 0) {
    warning("No stopped trials were found; choice-by-stop panels will be empty.")
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

  node_cols <- intersect(c("beta", "opportunity", "seed", "graph", "chosen_path", "node", "actual_reward"), names(dat))
  node_data <- unique(dat[, node_cols, drop = FALSE])
  node_data <- merge(
    node_data,
    trial_stop[, intersect(c("beta", "opportunity", "seed", "graph", "stop_timestep"), names(trial_stop)), drop = FALSE],
    by = intersect(c("beta", "opportunity", "seed", "graph"), names(node_data))
  )
  node_data$node <- suppressWarnings(as.numeric(node_data$node))
  node_data$chosen_path <- suppressWarnings(as.numeric(node_data$chosen_path))
  node_data$chose <- node_in_chosen_path(node_data$node, node_data$chosen_path)
  node_data$actual_reward <- suppressWarnings(as.numeric(node_data$actual_reward))
  choice_data <- node_data[
    !is.na(node_data$node) &
      !is.na(node_data$chose) &
      !is.na(node_data$stop_timestep) &
      !is.na(node_data$actual_reward),
    ,
    drop = FALSE
  ]
  choice_data$chose <- as.numeric(choice_data$chose)
  names(choice_data)[names(choice_data) == "actual_reward"] <- "reward"

  if (nrow(choice_data) == 0) {
    warning("No candidate rewards were found for stopped trials; choice-by-stop panels will be empty.")
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

  choice_summary <- aggregate(
    chose ~ beta + opportunity + stop_timestep + reward,
    data = choice_data,
    FUN = mean
  )
  names(choice_summary)[names(choice_summary) == "chose"] <- "p_choose"
  choice_counts <- aggregate(
    chose ~ beta + opportunity + stop_timestep + reward,
    data = choice_data,
    FUN = length
  )
  names(choice_counts)[names(choice_counts) == "chose"] <- "n"
  merge(
    choice_summary,
    choice_counts,
    by = c("beta", "opportunity", "stop_timestep", "reward")
  )[, c("beta", "opportunity", "stop_timestep", "reward", "p_choose", "n")]
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
    warning(sprintf(
      "No trials stopped at decision timestep %d; choice-vs-other panels will be empty.",
      stop_timestep
    ))
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
  node_data$chose <- node_in_chosen_path(node_data$node, node_data$chosen_path)
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
      !is.na(choice_other_data$other_reward) &
      !is.na(choice_other_data$chose),
    ,
    drop = FALSE
  ]
  choice_other_data$chose <- as.numeric(choice_other_data$chose)
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

empty_sequential_choice_summary <- function() {
  data.frame(
    beta = character(),
    opportunity = character(),
    decision_timestep = integer(),
    reward_t1 = numeric(),
    reward_t2 = numeric(),
    reward_current = numeric(),
    p_choose = numeric(),
    n = integer(),
    stringsAsFactors = FALSE
  )
}

build_sequential_choice_summary <- function(dat) {
  required_cols <- c(
    "chosen_path",
    "expanded_node_t1", "expanded_reward_t1",
    "expanded_node_t2", "expanded_reward_t2", "stop_t3"
  )
  missing_cols <- setdiff(required_cols, names(dat))
  if (length(missing_cols) > 0) {
    warning(sprintf(
      "Cannot build sequential choice summary. Missing columns: %s",
      paste(missing_cols, collapse = ", ")
    ))
    return(empty_sequential_choice_summary())
  }

  trial_id_cols <- intersect(c("beta", "opportunity", "seed", "graph"), names(dat))
  trial_cols <- unique(c(
    trial_id_cols,
    "chosen_path",
    grep("^expanded_node_t[0-9]+$", names(dat), value = TRUE),
    grep("^expanded_reward_t[0-9]+$", names(dat), value = TRUE),
    grep("^stop_t[0-9]+$", names(dat), value = TRUE)
  ))
  trial_data <- unique(dat[, trial_cols, drop = FALSE])

  numeric_cols <- grep("^expanded_(node|reward)_t[0-9]+$", names(trial_data), value = TRUE)
  for (col in numeric_cols) {
    trial_data[[col]] <- suppressWarnings(as.numeric(trial_data[[col]]))
  }
  trial_data$chosen_path <- suppressWarnings(as.numeric(trial_data$chosen_path))

  choice_parts <- list()

  t2_trials <- trial_data[
    !is.na(trial_data$expanded_reward_t1) &
      !is.na(trial_data$expanded_reward_t2) &
      as_logical_col(trial_data$stop_t3),
    ,
    drop = FALSE
  ]
  if (nrow(t2_trials) > 0) {
    t2_chose <- node_in_chosen_path(t2_trials$expanded_node_t2, t2_trials$chosen_path)
    choice_parts[[length(choice_parts) + 1]] <- data.frame(
      beta = t2_trials$beta,
      opportunity = t2_trials$opportunity,
      decision_timestep = 2L,
      reward_t1 = t2_trials$expanded_reward_t1,
      reward_t2 = NA_real_,
      reward_current = t2_trials$expanded_reward_t2,
      chose = as.numeric(t2_chose),
      stringsAsFactors = FALSE
    )
  }

  if (all(c("expanded_node_t3", "expanded_reward_t3") %in% names(trial_data))) {
    t3_trials <- trial_data[
      !is.na(trial_data$expanded_reward_t1) &
        !is.na(trial_data$expanded_reward_t2) &
        !is.na(trial_data$expanded_reward_t3),
      ,
      drop = FALSE
    ]
    if (nrow(t3_trials) > 0) {
      t3_chose <- node_in_chosen_path(t3_trials$expanded_node_t3, t3_trials$chosen_path)
      choice_parts[[length(choice_parts) + 1]] <- data.frame(
        beta = t3_trials$beta,
        opportunity = t3_trials$opportunity,
        decision_timestep = 3L,
        reward_t1 = t3_trials$expanded_reward_t1,
        reward_t2 = t3_trials$expanded_reward_t2,
        reward_current = t3_trials$expanded_reward_t3,
        chose = as.numeric(t3_chose),
        stringsAsFactors = FALSE
      )
    }
  }

  if (length(choice_parts) == 0) {
    warning("No sequential choice trials matched the requested conditioning.")
    return(empty_sequential_choice_summary())
  }

  choice_data <- do.call(rbind, choice_parts)
  choice_data <- choice_data[!is.na(choice_data$chose), , drop = FALSE]
  if (nrow(choice_data) == 0) {
    warning("No valid sequential choice labels were found.")
    return(empty_sequential_choice_summary())
  }

  summarize_choice_data <- function(part_data, group_cols) {
    if (nrow(part_data) == 0) return(empty_sequential_choice_summary())
    formula_text <- paste("chose ~", paste(group_cols, collapse = " + "))
    choice_summary <- aggregate(
      as.formula(formula_text),
      data = part_data,
      FUN = mean
    )
    names(choice_summary)[names(choice_summary) == "chose"] <- "p_choose"
    choice_counts <- aggregate(
      as.formula(formula_text),
      data = part_data,
      FUN = length
    )
    names(choice_counts)[names(choice_counts) == "chose"] <- "n"
    merge(choice_summary, choice_counts, by = group_cols)
  }

  t2_summary <- summarize_choice_data(
    choice_data[choice_data$decision_timestep == 2, , drop = FALSE],
    c("beta", "opportunity", "decision_timestep", "reward_t1", "reward_current")
  )
  if (nrow(t2_summary) > 0) {
    t2_summary$reward_t2 <- NA_real_
    t2_summary <- t2_summary[
      c("beta", "opportunity", "decision_timestep", "reward_t1", "reward_t2", "reward_current", "p_choose", "n")
    ]
  }

  t3_summary <- summarize_choice_data(
    choice_data[choice_data$decision_timestep == 3, , drop = FALSE],
    c("beta", "opportunity", "decision_timestep", "reward_t1", "reward_t2", "reward_current")
  )
  if (nrow(t3_summary) > 0) {
    t3_summary <- t3_summary[
      c("beta", "opportunity", "decision_timestep", "reward_t1", "reward_t2", "reward_current", "p_choose", "n")
    ]
  }

  do.call(rbind, list(t2_summary, t3_summary))
}

empty_disjoint2x2_path_choice_summary <- function() {
  data.frame(
    beta = character(),
    opportunity = character(),
    stop_timestep = integer(),
    current_path_reward = numeric(),
    other_path_reward = numeric(),
    p_choose = numeric(),
    n = integer(),
    stringsAsFactors = FALSE
  )
}

build_disjoint2x2_path_choice_summary <- function(dat) {
  required_cols <- c("chosen_path")
  expanded_node_cols <- grep("^expanded_node_t[0-9]+$", names(dat), value = TRUE)
  expanded_reward_cols <- grep("^expanded_reward_t[0-9]+$", names(dat), value = TRUE)
  stop_cols <- grep("^stop_t[0-9]+$", names(dat), value = TRUE)
  if (
    any(!required_cols %in% names(dat)) ||
      length(expanded_node_cols) == 0 ||
      length(expanded_reward_cols) == 0 ||
      length(stop_cols) == 0
  ) {
    warning(
      "Cannot build disjoint2x2 path-choice summary; expected chosen_path, expanded_node_t*, expanded_reward_t*, and stop_t* columns."
    )
    return(empty_disjoint2x2_path_choice_summary())
  }

  node_timesteps <- as.integer(sub("^expanded_node_t", "", expanded_node_cols))
  reward_timesteps <- as.integer(sub("^expanded_reward_t", "", expanded_reward_cols))
  common_timesteps <- sort(intersect(node_timesteps, reward_timesteps))
  if (length(common_timesteps) == 0) {
    warning("Cannot build disjoint2x2 path-choice summary; no matching expanded_node_t*/expanded_reward_t* timesteps.")
    return(empty_disjoint2x2_path_choice_summary())
  }

  trial_stop <- get_trial_stop_data(dat)
  trial_stop <- trial_stop[!is.na(trial_stop$stop_timestep), , drop = FALSE]
  if (nrow(trial_stop) == 0) {
    warning("No stopped trials found; disjoint2x2 path-choice plot will be empty.")
    return(empty_disjoint2x2_path_choice_summary())
  }

  trial_id_cols <- intersect(c("beta", "opportunity", "seed", "graph", "chosen_path"), names(dat))
  trial_cols <- unique(c(
    trial_id_cols,
    sprintf("expanded_node_t%d", common_timesteps),
    sprintf("expanded_reward_t%d", common_timesteps)
  ))
  trial_data <- unique(dat[, trial_cols, drop = FALSE])
  trial_data <- merge(
    trial_data,
    trial_stop[, intersect(c("beta", "opportunity", "seed", "graph", "stop_timestep"), names(trial_stop)), drop = FALSE],
    by = intersect(c("beta", "opportunity", "seed", "graph"), names(trial_data)),
    all.x = FALSE,
    all.y = FALSE
  )

  if (nrow(trial_data) == 0) {
    warning("No stopped disjoint2x2 trials matched expanded reward rows; path-choice plot will be empty.")
    return(empty_disjoint2x2_path_choice_summary())
  }

  trial_data$chosen_path <- suppressWarnings(as.numeric(trial_data$chosen_path))

  rows <- list()
  row_i <- 1
  for (i in seq_len(nrow(trial_data))) {
    stop_timestep <- suppressWarnings(as.integer(trial_data$stop_timestep[[i]]))
    if (is.na(stop_timestep)) {
      next
    }

    path_rewards <- c(NA_real_, NA_real_)
    path_observed <- c(FALSE, FALSE)
    for (timestep in common_timesteps) {
      if (timestep >= stop_timestep) {
        next
      }
      node_col <- sprintf("expanded_node_t%d", timestep)
      reward_col <- sprintf("expanded_reward_t%d", timestep)
      node_value <- suppressWarnings(as.numeric(trial_data[[node_col]][[i]]))
      reward_value <- suppressWarnings(as.numeric(trial_data[[reward_col]][[i]]))
      path_value <- disjoint_path_id(node_value)
      if (is.na(path_value) || is.na(reward_value) || path_value < 0 || path_value > 1) {
        next
      }
      path_idx <- as.integer(path_value) + 1L
      if (is.na(path_rewards[[path_idx]])) {
        path_rewards[[path_idx]] <- 0
      }
      path_rewards[[path_idx]] <- path_rewards[[path_idx]] + reward_value
      path_observed[[path_idx]] <- TRUE
    }

    if (!all(path_observed)) {
      next
    }

    for (candidate_path in 0:1) {
      other_path <- 1 - candidate_path
      rows[[row_i]] <- data.frame(
        beta = trial_data$beta[[i]],
        opportunity = trial_data$opportunity[[i]],
        stop_timestep = stop_timestep,
        current_path_reward = path_rewards[[candidate_path + 1L]],
        other_path_reward = path_rewards[[other_path + 1L]],
        chose = as.numeric(trial_data$chosen_path[[i]] == candidate_path),
        stringsAsFactors = FALSE
      )
      row_i <- row_i + 1
    }
  }

  if (length(rows) == 0) {
    warning("No stopped disjoint2x2 trials had observed rewards on both paths at selection time.")
    return(empty_disjoint2x2_path_choice_summary())
  }

  choice_data <- do.call(rbind, rows)
  choice_data <- choice_data[
    !is.na(choice_data$current_path_reward) &
      !is.na(choice_data$other_path_reward) &
      !is.na(choice_data$chose),
    ,
    drop = FALSE
  ]
  if (nrow(choice_data) == 0) {
    return(empty_disjoint2x2_path_choice_summary())
  }

  group_cols <- c(
    "beta",
    "opportunity",
    "stop_timestep",
    "current_path_reward",
    "other_path_reward"
  )
  choice_summary <- aggregate(
    as.formula(paste("chose ~", paste(group_cols, collapse = " + "))),
    data = choice_data,
    FUN = mean
  )
  names(choice_summary)[names(choice_summary) == "chose"] <- "p_choose"
  choice_counts <- aggregate(
    as.formula(paste("chose ~", paste(group_cols, collapse = " + "))),
    data = choice_data,
    FUN = length
  )
  names(choice_counts)[names(choice_counts) == "chose"] <- "n"

  merge(choice_summary, choice_counts, by = group_cols)[
    c(
      "beta",
      "opportunity",
      "stop_timestep",
      "current_path_reward",
      "other_path_reward",
      "p_choose",
      "n"
    )
  ]
}

kl_reward_summary <- build_kl_by_reward_all_summary(all_data)
kl_transition_heatmap_summary <- build_kl_transition_heatmap_summary(all_data)
reconstruction_accuracy_summary <- build_reconstruction_accuracy_summary(all_data)
deep_probe_accuracy_summary <- build_deep_probe_accuracy_summary(all_data)
deep_probe_confusion_summary <- build_deep_probe_confusion_summary(all_data)
deep_probe_signed_error_summary <- build_deep_probe_signed_error_summary(all_data)
deep_probe_path_context_signed_error_summary <- build_deep_probe_path_context_signed_error_summary(all_data)
deep_probe_t2_conditioned_summary <- build_deep_probe_t2_conditioned_summary(all_data)
choice_stop_summary <- build_choice_by_stop_summary(all_data)
choice_final_summary <- if (is_disjoint2x2) {
  build_disjoint2x2_path_choice_summary(all_data)
} else {
  build_sequential_choice_summary(all_data)
}

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
  warning("Both beta and opportunity have multiple values; using beta for color and keeping opportunity point/line styles fixed.")
}

color_levels <- if (identical(color_by, "opportunity")) opportunity_levels else beta_levels
color_level_labels <- if (identical(color_by, "opportunity")) {
  format_plot_values(color_levels)
} else {
  format_plot_values(color_levels)
}
palette_cols <- grDevices::hcl.colors(max(3, length(color_levels) + 2), palette = "Blues")
palette_cols <- palette_cols[seq_along(color_levels)]
if (identical(color_by, "opportunity")) {
  palette_cols <- rev(palette_cols)
}
color_cols <- setNames(palette_cols, color_levels)
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
opportunity_pch <- setNames(
  rep(19, length.out = length(opportunity_levels)),
  opportunity_levels
)
opportunity_lty <- setNames(
  rep(1, length.out = length(opportunity_levels)),
  opportunity_levels
)

expand_range <- function(x, pad = 0.5) {
  x_range <- suppressWarnings(range(x, finite = TRUE))
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
  par(xpd = old_xpd)
}

plot_reward_timestep_summary <- function(
  summary_data,
  value_col,
  ylab,
  main_prefix,
  empty_message,
  y_limits = NULL,
  timesteps = NULL
) {
  if (is.null(timesteps)) {
    timesteps <- sort(unique(summary_data$timestep))
  }
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
    "continue_probability_%s_lambda_%s_alpha_%s_beta_%s_opportunity_%s_expansion_%s_%s.pdf",
    input_type, lambda_arg, alpha_arg, beta_label, opportunity_label, expansion_label, tree_file_label
  )
)

continue_panel_mar <- c(4.5, 4.5, 4.2, 8)

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
      main = panel_title,
      cex.main = 0.9,
      xaxt = "n"
    )
    axis(1, at = c(0, 1))
    grid()
    text(
      0.5,
      0.5,
      sprintf(
        "No timestep %d decision exists\nafter reward_t%d",
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
    main = panel_title,
    cex.main = 0.9,
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

plot_initial_stop_panel <- function(opportunity_value = NULL) {
  panel_data <- initial_stop_summary
  if (!is.null(opportunity_value)) {
    panel_data <- panel_data[
      panel_data$opportunity == opportunity_value,
      ,
      drop = FALSE
    ]
  }

  x_param <- if (is.null(opportunity_value) && identical(color_by, "opportunity")) {
    "opportunity"
  } else {
    "beta"
  }
  x_levels <- if (identical(x_param, "opportunity")) opportunity_levels else beta_levels
  x_labels <- format_plot_values(x_levels)
  x_positions <- setNames(seq_along(x_levels), x_levels)
  opportunity_title <- if (is.null(opportunity_value)) {
    "all opportunities"
  } else {
    sprintf("opportunity %s", opportunity_value)
  }
  panel_title <- sprintf("%s\nContinue at timestep 1 before any reward", opportunity_title)

  plot(
    NA,
    xlim = c(0.5, max(1.5, length(x_levels) + 0.5)),
    ylim = c(0, 1),
    xlab = if (identical(x_param, "opportunity")) "Opportunity cost" else "Beta",
    ylab = "P(continue at timestep 1)",
    main = panel_title,
    cex.main = 0.9,
    xaxt = "n"
  )
  axis(1, at = seq_along(x_levels), labels = x_labels)
  grid()

  if (nrow(panel_data) == 0) {
    text(mean(par("usr")[1:2]), 0.5, "No timestep 1 decisions", cex = 0.9)
    return(invisible(NULL))
  }

  point_x <- x_positions[as.character(panel_data[[x_param]])]
  point_col <- mapply(series_color, panel_data$beta, panel_data$opportunity)
  point_pch <- opportunity_pch[as.character(panel_data$opportunity)]
  points(
    point_x,
    panel_data$p_continue_initial,
    pch = point_pch,
    col = point_col,
    cex = 1.4,
    lwd = 2
  )
}

plot_continue_t3_conditioned_heatmaps <- function(summary_data) {
  panel_keys <- expand.grid(
    beta = beta_levels,
    opportunity = opportunity_levels,
    KEEP.OUT.ATTRS = FALSE,
    stringsAsFactors = FALSE
  )

  if (nrow(panel_keys) == 0) {
    panel_keys <- data.frame(
      beta = NA_character_,
      opportunity = NA_character_,
      stringsAsFactors = FALSE
    )
  }

  reward_levels <- sort(unique(c(
    suppressWarnings(as.numeric(summary_data$reward_t1)),
    suppressWarnings(as.numeric(summary_data$reward_t2)),
    suppressWarnings(as.numeric(all_data$actual_reward))
  )))
  reward_levels <- reward_levels[!is.na(reward_levels)]
  if (length(reward_levels) == 0) {
    reward_levels <- sort(unique(c(summary_data$reward_t1, summary_data$reward_t2)))
    reward_levels <- reward_levels[!is.na(reward_levels)]
  }
  if (length(reward_levels) == 0) {
    reward_levels <- c(0, 1)
  }

  n_cols <- min(3, nrow(panel_keys))
  n_rows <- ceiling(nrow(panel_keys) / n_cols)
  heat_cols <- grDevices::hcl.colors(64, palette = "Blues")
  old_par <- par(mfrow = c(n_rows, n_cols), mar = c(4.5, 4.5, 3, 1))

  for (panel_i in seq_len(nrow(panel_keys))) {
    beta_value <- panel_keys$beta[[panel_i]]
    opportunity_value <- panel_keys$opportunity[[panel_i]]
    panel_data <- summary_data[
      summary_data$beta == beta_value &
        summary_data$opportunity == opportunity_value,
      ,
      drop = FALSE
    ]

    panel_title <- sprintf(
      "P(continue at t3) | beta %s, opportunity %s",
      beta_value,
      opportunity_value
    )

    if (nrow(panel_data) == 0) {
      plot(
        NA,
        xlim = expand_range(reward_levels, pad = 0.1),
        ylim = expand_range(reward_levels, pad = 0.1),
        xlab = "Reward observed at timestep 1",
        ylab = "Reward observed at timestep 2",
        main = panel_title,
        xaxt = "n",
        yaxt = "n"
      )
      axis(1, at = reward_levels)
      axis(2, at = reward_levels)
      grid()
      text(mean(par("usr")[1:2]), mean(par("usr")[3:4]), "No t3 decisions", cex = 0.9)
      next
    }

    z <- matrix(NA_real_, nrow = length(reward_levels), ncol = length(reward_levels))
    for (row_i in seq_len(nrow(panel_data))) {
      x_i <- match(panel_data$reward_t1[[row_i]], reward_levels)
      y_i <- match(panel_data$reward_t2[[row_i]], reward_levels)
      if (!is.na(x_i) && !is.na(y_i)) {
        z[x_i, y_i] <- panel_data$p_continue_t3[[row_i]]
      }
    }

    image(
      reward_levels,
      reward_levels,
      z,
      zlim = c(0, 1),
      col = heat_cols,
      xlab = "Reward observed at timestep 1",
      ylab = "Reward observed at timestep 2",
      main = panel_title,
      xaxt = "n",
      yaxt = "n"
    )
    axis(1, at = reward_levels)
    axis(2, at = reward_levels)
    grid()
    for (row_i in seq_len(nrow(panel_data))) {
      text(
        panel_data$reward_t1[[row_i]],
        panel_data$reward_t2[[row_i]],
        labels = format(signif(panel_data$p_continue_t3[[row_i]], 2), trim = TRUE),
        cex = 0.7
      )
    }
  }

  par(old_par)
}

plot_continue_t3_max_conditioned_panel <- function(summary_data) {
  panel_title <- "P(continue at t3) by max reward observed at t1-t2"
  reward_levels <- sort(unique(c(
    suppressWarnings(as.numeric(summary_data$max_reward_observed)),
    suppressWarnings(as.numeric(all_data$actual_reward))
  )))
  reward_levels <- reward_levels[!is.na(reward_levels)]
  if (length(reward_levels) == 0) {
    reward_levels <- c(0, 1)
  }

  plot(
    NA,
    xlim = expand_range(reward_levels, pad = 0.1),
    ylim = c(0, 1),
    xlab = "Max reward observed by timestep 2",
    ylab = "P(continue at timestep 3)",
    main = panel_title,
    xaxt = "n",
    cex.main = 0.9
  )
  axis(1, at = reward_levels)
  grid()

  if (nrow(summary_data) == 0) {
    text(mean(par("usr")[1:2]), 0.5, "No t3 decisions", cex = 0.9)
    return(invisible(NULL))
  }

  for (beta_value in beta_levels) {
    for (opportunity_value in opportunity_levels) {
      series_data <- summary_data[
        summary_data$beta == beta_value &
          summary_data$opportunity == opportunity_value,
        ,
        drop = FALSE
      ]
      if (nrow(series_data) == 0) {
        next
      }
      series_data <- series_data[order(series_data$max_reward_observed), , drop = FALSE]
      lines(
        series_data$max_reward_observed,
        series_data$p_continue_t3,
        type = "b",
        pch = opportunity_pch[[as.character(opportunity_value)]],
        col = series_color(beta_value, opportunity_value),
        lwd = 2
      )
    }
  }
  plot_parameter_legend()
}

plot_continue_t4_max_conditioned_panel <- function(summary_data) {
  panel_title <- "P(continue at t4) by max reward observed at t1-t3"
  reward_levels <- sort(unique(c(
    suppressWarnings(as.numeric(summary_data$max_reward_observed)),
    suppressWarnings(as.numeric(all_data$actual_reward))
  )))
  reward_levels <- reward_levels[!is.na(reward_levels)]
  if (length(reward_levels) == 0) {
    reward_levels <- c(0, 1)
  }

  plot(
    NA,
    xlim = expand_range(reward_levels, pad = 0.1),
    ylim = c(0, 1),
    xlab = "Max reward observed by timestep 3",
    ylab = "P(continue at timestep 4)",
    main = panel_title,
    xaxt = "n",
    cex.main = 0.9
  )
  axis(1, at = reward_levels)
  grid()

  if (nrow(summary_data) == 0) {
    text(mean(par("usr")[1:2]), 0.5, "No t4 decisions", cex = 0.9)
    return(invisible(NULL))
  }

  for (beta_value in beta_levels) {
    for (opportunity_value in opportunity_levels) {
      series_data <- summary_data[
        summary_data$beta == beta_value &
          summary_data$opportunity == opportunity_value,
        ,
        drop = FALSE
      ]
      if (nrow(series_data) == 0) {
        next
      }
      series_data <- series_data[order(series_data$max_reward_observed), , drop = FALSE]
      lines(
        series_data$max_reward_observed,
        series_data$p_continue_t4,
        type = "b",
        pch = opportunity_pch[[as.character(opportunity_value)]],
        col = series_color(beta_value, opportunity_value),
        lwd = 2
      )
    }
  }
  plot_parameter_legend()
}

plot_disjoint2x2_t2_path_continue_panel <- function(summary_data) {
  panel_title <- "P(observe path at t2 | continue at t2)"
  reward_levels <- sort(unique(c(
    suppressWarnings(as.numeric(summary_data$reward_t1)),
    suppressWarnings(as.numeric(all_data$actual_reward))
  )))
  reward_levels <- reward_levels[!is.na(reward_levels)]
  if (length(reward_levels) == 0) {
    reward_levels <- c(0, 1)
  }

  plot(
    NA,
    xlim = expand_range(reward_levels, pad = 0.1),
    ylim = c(0, 1),
    xlab = "Reward observed at timestep 1",
    ylab = "Probability",
    main = panel_title,
    xaxt = "n",
    cex.main = 0.9
  )
  axis(1, at = reward_levels)
  grid()

  if (nrow(summary_data) == 0) {
    text(mean(par("usr")[1:2]), 0.5, "No continued t2 path choices", cex = 0.9)
    return(invisible(NULL))
  }

  for (beta_value in beta_levels) {
    for (opportunity_value in opportunity_levels) {
      series_data <- summary_data[
        summary_data$beta == beta_value &
          summary_data$opportunity == opportunity_value,
        ,
        drop = FALSE
      ]
      if (nrow(series_data) == 0) {
        next
      }
      series_data <- series_data[order(series_data$reward_t1), , drop = FALSE]
      line_col <- series_color(beta_value, opportunity_value)
      lines(
        series_data$reward_t1,
        series_data$p_current_path_t2,
        type = "b",
        pch = opportunity_pch[[as.character(opportunity_value)]],
        col = line_col,
        lwd = 2,
        lty = 1
      )
      lines(
        series_data$reward_t1,
        series_data$p_different_path_t2,
        type = "b",
        pch = opportunity_pch[[as.character(opportunity_value)]],
        col = line_col,
        lwd = 2,
        lty = 2
      )
    }
  }

  legend(
    "bottomright",
    legend = c("current path", "different path"),
    lty = c(1, 2),
    lwd = 2,
    col = "gray20",
    bty = "n",
    cex = 0.85
  )
  plot_parameter_legend()
}

plot_disjoint2x2_t3_path_continue_heatmaps <- function(summary_data) {
  panel_keys <- expand.grid(
    beta = beta_levels,
    opportunity = opportunity_levels,
    KEEP.OUT.ATTRS = FALSE,
    stringsAsFactors = FALSE
  )

  if (nrow(panel_keys) == 0) {
    panel_keys <- data.frame(
      beta = NA_character_,
      opportunity = NA_character_,
      stringsAsFactors = FALSE
    )
  }

  reward_levels <- sort(unique(c(
    suppressWarnings(as.numeric(summary_data$reward_t1)),
    suppressWarnings(as.numeric(summary_data$reward_t2)),
    suppressWarnings(as.numeric(all_data$actual_reward))
  )))
  reward_levels <- reward_levels[!is.na(reward_levels)]
  if (length(reward_levels) == 0) {
    reward_levels <- c(0, 1)
  }

  n_cols <- min(3, nrow(panel_keys))
  n_rows <- ceiling(nrow(panel_keys) / n_cols)
  heat_cols <- grDevices::hcl.colors(64, palette = "Blues")
  old_par <- par(mfrow = c(n_rows, n_cols), mar = c(4.5, 4.5, 3.4, 1))

  for (panel_i in seq_len(nrow(panel_keys))) {
    beta_value <- panel_keys$beta[[panel_i]]
    opportunity_value <- panel_keys$opportunity[[panel_i]]
    panel_data <- summary_data[
      summary_data$beta == beta_value &
        summary_data$opportunity == opportunity_value,
      ,
      drop = FALSE
    ]

    panel_title <- sprintf(
      "T3 path choice | beta %s, opportunity %s\nlabels: current/different path",
      beta_value,
      opportunity_value
    )

    if (nrow(panel_data) == 0) {
      plot(
        NA,
        xlim = expand_range(reward_levels, pad = 0.1),
        ylim = expand_range(reward_levels, pad = 0.1),
        xlab = "Reward observed at timestep 1",
        ylab = "Reward observed at timestep 2",
        main = panel_title,
        xaxt = "n",
        yaxt = "n",
        cex.main = 0.85
      )
      axis(1, at = reward_levels)
      axis(2, at = reward_levels)
      grid()
      text(mean(par("usr")[1:2]), mean(par("usr")[3:4]), "No continued t3 path choices", cex = 0.9)
      next
    }

    z <- matrix(NA_real_, nrow = length(reward_levels), ncol = length(reward_levels))
    for (row_i in seq_len(nrow(panel_data))) {
      x_i <- match(panel_data$reward_t1[[row_i]], reward_levels)
      y_i <- match(panel_data$reward_t2[[row_i]], reward_levels)
      if (!is.na(x_i) && !is.na(y_i)) {
        z[x_i, y_i] <- panel_data$p_current_path_t3[[row_i]]
      }
    }

    image(
      reward_levels,
      reward_levels,
      z,
      zlim = c(0, 1),
      col = heat_cols,
      xlab = "Reward observed at timestep 1",
      ylab = "Reward observed at timestep 2",
      main = panel_title,
      xaxt = "n",
      yaxt = "n",
      cex.main = 0.85
    )
    axis(1, at = reward_levels)
    axis(2, at = reward_levels)
    grid()
    for (row_i in seq_len(nrow(panel_data))) {
      text(
        panel_data$reward_t1[[row_i]],
        panel_data$reward_t2[[row_i]],
        labels = sprintf(
          "%s/%s",
          format(signif(panel_data$p_current_path_t3[[row_i]], 2), trim = TRUE),
          format(signif(panel_data$p_different_path_t3[[row_i]], 2), trim = TRUE)
        ),
        cex = 0.62
      )
    }
  }

  par(old_par)
}

plot_disjoint2x2_continue_by_path_reward_panel <- function(
    summary_data,
    value_col,
    x_col,
    x_label,
    y_label,
    panel_title,
    empty_label) {
  reward_levels <- sort(unique(c(
    suppressWarnings(as.numeric(summary_data[[x_col]])),
    suppressWarnings(as.numeric(all_data$actual_reward))
  )))
  reward_levels <- reward_levels[!is.na(reward_levels)]
  if (length(reward_levels) == 0) {
    reward_levels <- c(0, 1)
  }

  plot(
    NA,
    xlim = expand_range(reward_levels, pad = 0.1),
    ylim = c(0, 1),
    xlab = x_label,
    ylab = y_label,
    main = panel_title,
    xaxt = "n",
    cex.main = 0.9
  )
  axis(1, at = reward_levels)
  grid()

  if (nrow(summary_data) == 0 || !value_col %in% names(summary_data)) {
    text(mean(par("usr")[1:2]), 0.5, empty_label, cex = 0.9)
    return(invisible(NULL))
  }

  for (beta_value in beta_levels) {
    for (opportunity_value in opportunity_levels) {
      series_data <- summary_data[
        summary_data$beta == beta_value &
          summary_data$opportunity == opportunity_value,
        ,
        drop = FALSE
      ]
      if (nrow(series_data) == 0) {
        next
      }
      series_data <- series_data[order(series_data[[x_col]]), , drop = FALSE]
      lines(
        series_data[[x_col]],
        series_data[[value_col]],
        type = "b",
        pch = opportunity_pch[[as.character(opportunity_value)]],
        col = series_color(beta_value, opportunity_value),
        lwd = 2
      )
    }
  }
  plot_parameter_legend()
}

plot_disjoint2x2_t3_same_path_reward_panel <- function(summary_data) {
  plot_disjoint2x2_continue_by_path_reward_panel(
    summary_data = summary_data,
    value_col = "p_continue_t3",
    x_col = "observed_path_reward",
    x_label = "Observed path reward at timesteps 1-2",
    y_label = "P(continue at timestep 3)",
    panel_title = "P(continue at t3) after observing same path twice",
    empty_label = "No same-path t3 decisions"
  )
}

plot_disjoint2x2_t4_observed_path_reward_panel <- function(summary_data) {
  plot_disjoint2x2_continue_by_path_reward_panel(
    summary_data = summary_data,
    value_col = "p_continue_t4",
    x_col = "observed_path_reward",
    x_label = "Fully observed path reward by timestep 3",
    y_label = "P(continue at timestep 4)",
    panel_title = "P(continue at t4) by fully observed path reward",
    empty_label = "No t4 decisions with a fully observed path"
  )
}

plot_disjoint2x2_path_value_diff_continue_panel <- function(
    summary_data,
    decision_timestep,
    opportunity_value = NULL) {
  if (is.null(decision_timestep)) {
    if (nrow(summary_data) > 0) {
      weighted_data <- summary_data
      weighted_data$continue_n <- weighted_data$p_continue * weighted_data$n
      continue_summary <- aggregate(
        continue_n ~ beta + opportunity + value_diff,
        data = weighted_data,
        FUN = sum
      )
      continue_counts <- aggregate(
        n ~ beta + opportunity + value_diff,
        data = weighted_data,
        FUN = sum
      )
      panel_summary_data <- merge(
        continue_summary,
        continue_counts,
        by = c("beta", "opportunity", "value_diff")
      )
      panel_summary_data$p_continue <- panel_summary_data$continue_n / panel_summary_data$n
    } else {
      panel_summary_data <- summary_data
    }
  } else {
    panel_summary_data <- summary_data
  }

  value_levels <- sort(unique(suppressWarnings(as.numeric(panel_summary_data$value_diff))))
  value_levels <- value_levels[!is.na(value_levels)]
  if (length(value_levels) == 0) {
    value_levels <- c(-1, 1)
  }

  if (is.null(opportunity_value)) {
    if (is.null(decision_timestep)) {
      panel_data <- panel_summary_data
      panel_title <- "all opportunities\nContinue across all timesteps by current - other path value"
    } else {
      panel_data <- panel_summary_data[
        panel_summary_data$decision_timestep == decision_timestep,
        ,
        drop = FALSE
      ]
      panel_title <- sprintf(
        "all opportunities\nContinue at timestep %d by current - other path value",
        decision_timestep
      )
    }
  } else {
    if (is.null(decision_timestep)) {
      panel_data <- panel_summary_data[
        panel_summary_data$opportunity == opportunity_value,
        ,
        drop = FALSE
      ]
      panel_title <- sprintf(
        "opportunity %s\nContinue across all timesteps by current - other path value",
        opportunity_value
      )
    } else {
      panel_data <- panel_summary_data[
        panel_summary_data$decision_timestep == decision_timestep &
          panel_summary_data$opportunity == opportunity_value,
        ,
        drop = FALSE
      ]
      panel_title <- sprintf(
        "opportunity %s\nContinue at timestep %d by current - other path value",
        opportunity_value,
        decision_timestep
      )
    }
  }

  plot(
    NA,
    xlim = expand_range(value_levels, pad = 0.1),
    ylim = c(0, 1),
    xlab = "Observed value(current path) - observed value(other path)",
    ylab = "P(continue at current timestep)",
    main = panel_title,
    xaxt = "n",
    cex.main = 0.85
  )
  axis(1, at = value_levels)
  grid()

  if (nrow(panel_data) == 0) {
    text(mean(par("usr")[1:2]), 0.5, "No decisions with observed path-value differences", cex = 0.85)
    return(invisible(NULL))
  }

  for (beta_value in beta_levels) {
    for (opp_value in opportunity_levels) {
      series_data <- panel_data[
        panel_data$beta == beta_value &
          panel_data$opportunity == opp_value,
        ,
        drop = FALSE
      ]
      if (nrow(series_data) == 0) {
        next
      }
      series_data <- series_data[order(series_data$value_diff), , drop = FALSE]
      lines(
        series_data$value_diff,
        series_data$p_continue,
        type = "b",
        pch = opportunity_pch[[as.character(opp_value)]],
        lty = opportunity_lty[[as.character(opp_value)]],
        col = series_color(beta_value, opp_value),
        lwd = 2
      )
    }
  }
}

plot_disjoint2x2_path_value_diff_continue_page <- function(summary_data) {
  decision_timesteps <- sort(unique(suppressWarnings(as.integer(summary_data$decision_timestep))))
  decision_timesteps <- decision_timesteps[!is.na(decision_timesteps)]
  if (length(decision_timesteps) == 0) {
    decision_timesteps <- 2:min(4, tree_size)
  }
  panel_timesteps <- c(NA_integer_, decision_timesteps)

  n_cols <- max(stop_panel_count, length(panel_timesteps))
  if (identical(color_by, "opportunity")) {
    old_par <- par(mfrow = c(1, n_cols), mar = continue_panel_mar)
    for (decision_timestep in panel_timesteps) {
      if (is.na(decision_timestep)) {
        plot_disjoint2x2_path_value_diff_continue_panel(summary_data, NULL)
      } else {
        plot_disjoint2x2_path_value_diff_continue_panel(summary_data, decision_timestep)
      }
    }
    blank_count <- n_cols - length(panel_timesteps)
    if (blank_count > 0) {
      for (blank_i in seq_len(blank_count)) {
        plot.new()
      }
    }
  } else {
    old_par <- par(
      mfrow = c(length(opportunity_levels), n_cols),
      mar = continue_panel_mar
    )
    for (opportunity_value in opportunity_levels) {
      for (decision_timestep in panel_timesteps) {
        if (is.na(decision_timestep)) {
          plot_disjoint2x2_path_value_diff_continue_panel(
            summary_data,
            NULL,
            opportunity_value
          )
        } else {
          plot_disjoint2x2_path_value_diff_continue_panel(
            summary_data,
            decision_timestep,
            opportunity_value
          )
        }
      }
      blank_count <- n_cols - length(panel_timesteps)
      if (blank_count > 0) {
        for (blank_i in seq_len(blank_count)) {
          plot.new()
        }
      }
    }
  }
  plot_parameter_legend()
  par(old_par)
}

plot_disjoint2x2_reward_pair_heatmaps <- function(
    summary_data,
    value_col,
    panel_title_prefix,
    empty_label,
    x_label = "Reward observed at timestep 1",
    y_label = "Reward observed at timestep 2") {
  panel_keys <- expand.grid(
    beta = beta_levels,
    opportunity = opportunity_levels,
    KEEP.OUT.ATTRS = FALSE,
    stringsAsFactors = FALSE
  )

  if (nrow(panel_keys) == 0) {
    panel_keys <- data.frame(
      beta = NA_character_,
      opportunity = NA_character_,
      stringsAsFactors = FALSE
    )
  }

  reward_levels <- sort(unique(c(
    suppressWarnings(as.numeric(summary_data$reward_t1)),
    suppressWarnings(as.numeric(summary_data$reward_t2)),
    suppressWarnings(as.numeric(all_data$actual_reward))
  )))
  reward_levels <- reward_levels[!is.na(reward_levels)]
  if (length(reward_levels) == 0) {
    reward_levels <- c(0, 1)
  }

  n_cols <- min(3, nrow(panel_keys))
  n_rows <- ceiling(nrow(panel_keys) / n_cols)
  heat_cols <- grDevices::hcl.colors(64, palette = "Blues")
  old_par <- par(mfrow = c(n_rows, n_cols), mar = c(4.5, 4.5, 3.4, 1))

  for (panel_i in seq_len(nrow(panel_keys))) {
    beta_value <- panel_keys$beta[[panel_i]]
    opportunity_value <- panel_keys$opportunity[[panel_i]]
    panel_data <- summary_data[
      summary_data$beta == beta_value &
        summary_data$opportunity == opportunity_value,
      ,
      drop = FALSE
    ]

    panel_title <- sprintf(
      "%s | beta %s, opportunity %s",
      panel_title_prefix,
      beta_value,
      opportunity_value
    )

    if (nrow(panel_data) == 0 || !value_col %in% names(panel_data)) {
      plot(
        NA,
        xlim = expand_range(reward_levels, pad = 0.1),
        ylim = expand_range(reward_levels, pad = 0.1),
        xlab = x_label,
        ylab = y_label,
        main = panel_title,
        xaxt = "n",
        yaxt = "n",
        cex.main = 0.85
      )
      axis(1, at = reward_levels)
      axis(2, at = reward_levels)
      grid()
      text(mean(par("usr")[1:2]), mean(par("usr")[3:4]), empty_label, cex = 0.9)
      next
    }

    z <- matrix(NA_real_, nrow = length(reward_levels), ncol = length(reward_levels))
    for (row_i in seq_len(nrow(panel_data))) {
      x_i <- match(panel_data$reward_t1[[row_i]], reward_levels)
      y_i <- match(panel_data$reward_t2[[row_i]], reward_levels)
      if (!is.na(x_i) && !is.na(y_i)) {
        z[x_i, y_i] <- panel_data[[value_col]][[row_i]]
      }
    }

    image(
      reward_levels,
      reward_levels,
      z,
      zlim = c(0, 1),
      col = heat_cols,
      xlab = x_label,
      ylab = y_label,
      main = panel_title,
      xaxt = "n",
      yaxt = "n",
      cex.main = 0.85
    )
    axis(1, at = reward_levels)
    axis(2, at = reward_levels)
    grid()
    for (row_i in seq_len(nrow(panel_data))) {
      text(
        panel_data$reward_t1[[row_i]],
        panel_data$reward_t2[[row_i]],
        labels = format(signif(panel_data[[value_col]][[row_i]], 2), trim = TRUE),
        cex = 0.62
      )
    }
  }

  par(old_par)
}

plot_disjoint2x2_t3_different_path_reward_heatmaps <- function(summary_data) {
  plot_disjoint2x2_reward_pair_heatmaps(
    summary_data = summary_data,
    value_col = "p_continue_t3",
    panel_title_prefix = "P(continue at t3) after observing different paths",
    empty_label = "No different-path t3 decisions"
  )
}

plot_disjoint2x2_t3_better_path_continue_heatmaps <- function(summary_data) {
  plot_disjoint2x2_reward_pair_heatmaps(
    summary_data = summary_data,
    value_col = "p_continue_better_path_t3",
    panel_title_prefix = "P(observe better path at t3 | continue)",
    empty_label = "No unequal-reward continued t3 decisions"
  )
}

plot_action_probability_panel <- function(
    panel_data,
    x_col,
    y_cols,
    action_labels,
    action_colors,
    action_ltys,
    action_pchs,
    x_label,
    panel_title,
    empty_label) {
  x_values <- sort(unique(suppressWarnings(as.numeric(panel_data[[x_col]]))))
  x_values <- x_values[!is.na(x_values)]
  if (length(x_values) == 0) {
    x_values <- sort(unique(suppressWarnings(as.numeric(all_data$actual_reward))))
    x_values <- x_values[!is.na(x_values)]
  }
  if (length(x_values) == 0) {
    x_values <- c(0, 1)
  }

  plot(
    NA,
    xlim = expand_range(x_values, pad = 0.1),
    ylim = c(0, 1),
    xlab = x_label,
    ylab = "Probability",
    main = panel_title,
    xaxt = "n",
    cex.main = 0.82
  )
  axis(1, at = x_values)
  grid()

  if (nrow(panel_data) == 0) {
    text(mean(par("usr")[1:2]), 0.5, empty_label, cex = 0.85)
    return(invisible(NULL))
  }

  panel_data <- panel_data[order(panel_data[[x_col]]), , drop = FALSE]
  for (i in seq_along(y_cols)) {
    y_col <- y_cols[[i]]
    if (!y_col %in% names(panel_data)) {
      next
    }
    lines(
      panel_data[[x_col]],
      panel_data[[y_col]],
      type = "b",
      col = action_colors[[i]],
      lty = action_ltys[[i]],
      pch = action_pchs[[i]],
      lwd = 2
    )
  }
  legend(
    "topright",
    legend = action_labels,
    col = action_colors,
    lty = action_ltys,
    pch = action_pchs,
    lwd = 2,
    bty = "n",
    cex = 0.72
  )
}

legacy6_action_panel_keys <- function(summary_data, include_decision = TRUE, include_node_type = FALSE) {
  decision_levels <- sort(unique(suppressWarnings(as.integer(summary_data$decision_timestep))))
  decision_levels <- decision_levels[!is.na(decision_levels)]
  if (length(decision_levels) == 0) {
    decision_levels <- 1:min(6, tree_size)
  }
  node_type_levels <- if (include_node_type) {
    node_types <- sort(unique(as.character(summary_data$observed_node_type)))
    node_types <- node_types[nzchar(node_types) & !is.na(node_types)]
    if (length(node_types) == 0) c("middle", "leaf") else node_types
  } else {
    character()
  }

  panel_keys <- if (include_node_type) {
    expand.grid(
      beta = beta_levels,
      opportunity = opportunity_levels,
      decision_timestep = decision_levels,
      observed_node_type = node_type_levels,
      KEEP.OUT.ATTRS = FALSE,
      stringsAsFactors = FALSE
    )
  } else if (include_decision) {
    expand.grid(
      beta = beta_levels,
      opportunity = opportunity_levels,
      decision_timestep = decision_levels,
      KEEP.OUT.ATTRS = FALSE,
      stringsAsFactors = FALSE
    )
  } else {
    expand.grid(
      beta = beta_levels,
      opportunity = opportunity_levels,
      KEEP.OUT.ATTRS = FALSE,
      stringsAsFactors = FALSE
    )
  }
  if (nrow(panel_keys) == 0) {
    panel_keys <- data.frame(
      beta = if (length(beta_levels) > 0) beta_levels[[1]] else NA_character_,
      opportunity = if (length(opportunity_levels) > 0) opportunity_levels[[1]] else NA_character_,
      stringsAsFactors = FALSE
    )
    if (include_decision) {
      panel_keys$decision_timestep <- NA_integer_
    }
    if (include_node_type) {
      panel_keys$observed_node_type <- "middle"
    }
  }
  panel_keys
}

plot_legacy6_selected_node_pages <- function(summary_data) {
  decision_timesteps <- sort(unique(suppressWarnings(as.integer(summary_data$decision_timestep))))
  decision_timesteps <- decision_timesteps[!is.na(decision_timesteps)]
  if (length(decision_timesteps) == 0) {
    decision_timesteps <- 1:min(6, tree_size)
  }
  panel_keys <- expand.grid(
    beta = beta_levels,
    opportunity = opportunity_levels,
    KEEP.OUT.ATTRS = FALSE,
    stringsAsFactors = FALSE
  )
  if (nrow(panel_keys) == 0) {
    panel_keys <- data.frame(
      beta = if (length(beta_levels) > 0) beta_levels[[1]] else NA_character_,
      opportunity = if (length(opportunity_levels) > 0) opportunity_levels[[1]] else NA_character_,
      stringsAsFactors = FALSE
    )
  }

  for (decision_timestep in decision_timesteps) {
    plot_fixed_continue_grid_panels(
      panel_keys,
      function(beta_value, opportunity_value) {
        panel_data <- summary_data[
          summary_data$beta == beta_value &
            summary_data$opportunity == opportunity_value &
            summary_data$decision_timestep == decision_timestep,
          ,
          drop = FALSE
        ]
        x_values <- seq_len(6)
        plot(
          NA,
          xlim = expand_range(x_values, pad = 0.1),
          ylim = c(0, 1),
          xlab = "Node index selected if continue",
          ylab = "P(select node | continue)",
          main = sprintf(
            "Decision t%d | beta %s, opportunity %s",
            decision_timestep,
            beta_value,
            opportunity_value
          ),
          xaxt = "n",
          cex.main = 0.82
        )
        axis(1, at = x_values)
        grid()
        if (nrow(panel_data) == 0) {
          text(mean(par("usr")[1:2]), 0.5, "No continued decisions", cex = 0.85)
          return(invisible(NULL))
        }
        panel_data <- panel_data[order(panel_data$node_index), , drop = FALSE]
        lines(
          panel_data$node_index,
          panel_data$p_selected_given_continue,
          type = "b",
          col = "#2c7fb8",
          pch = 16,
          lwd = 2
        )
      }
    )
  }
}

plot_legacy6_path_action_by_reward_pages <- function(summary_data) {
  decision_timesteps <- sort(unique(suppressWarnings(as.integer(summary_data$decision_timestep))))
  decision_timesteps <- decision_timesteps[!is.na(decision_timesteps)]
  if (length(decision_timesteps) == 0) {
    decision_timesteps <- 2:min(6, tree_size)
  }
  node_type_levels <- sort(unique(as.character(summary_data$observed_node_type)))
  node_type_levels <- node_type_levels[nzchar(node_type_levels) & !is.na(node_type_levels)]
  if (length(node_type_levels) == 0) {
    node_type_levels <- c("middle", "leaf")
  }
  panel_keys <- expand.grid(
    beta = beta_levels,
    opportunity = opportunity_levels,
    KEEP.OUT.ATTRS = FALSE,
    stringsAsFactors = FALSE
  )
  if (nrow(panel_keys) == 0) {
    panel_keys <- data.frame(
      beta = if (length(beta_levels) > 0) beta_levels[[1]] else NA_character_,
      opportunity = if (length(opportunity_levels) > 0) opportunity_levels[[1]] else NA_character_,
      stringsAsFactors = FALSE
    )
  }

  for (decision_timestep in decision_timesteps) {
    for (node_type in node_type_levels) {
      plot_fixed_continue_grid_panels(
        panel_keys,
        function(beta_value, opportunity_value) {
          panel_data <- summary_data[
            summary_data$beta == beta_value &
              summary_data$opportunity == opportunity_value &
              summary_data$decision_timestep == decision_timestep &
              summary_data$observed_node_type == node_type,
            ,
            drop = FALSE
          ]
          plot_action_probability_panel(
            panel_data = panel_data,
            x_col = "observed_reward",
            y_cols = c("p_continue_current_path", "p_continue_different_path", "p_stop"),
            action_labels = c("continue current path", "continue different path", "stop"),
            action_colors = c("#2c7fb8", "#41ab5d", "#d95f0e"),
            action_ltys = c(1, 2, 1),
            action_pchs = c(16, 17, 15),
            x_label = sprintf("Reward at previous timestep (%s node)", node_type),
            panel_title = sprintf(
              "%s observed, decision t%d | beta %s, opportunity %s",
              node_type,
              decision_timestep,
              beta_value,
              opportunity_value
            ),
            empty_label = sprintf("No decisions after %s node rewards", node_type)
          )
        }
      )
    }
  }
}

plot_legacy6_value_continue_panel <- function(
    summary_data,
    decision_timestep,
    title_prefix,
    x_label,
    empty_label,
    opportunity_value = NULL) {
  value_levels <- sort(unique(suppressWarnings(as.numeric(summary_data$path_value))))
  value_levels <- value_levels[!is.na(value_levels)]
  if (length(value_levels) == 0) {
    value_levels <- c(-8, 8)
  }

  if (is.null(opportunity_value)) {
    panel_data <- summary_data[
      summary_data$decision_timestep == decision_timestep,
      ,
      drop = FALSE
    ]
    panel_title <- sprintf(
      "all opportunities\n%s, decision t%d",
      title_prefix,
      decision_timestep
    )
  } else {
    panel_data <- summary_data[
      summary_data$decision_timestep == decision_timestep &
        summary_data$opportunity == opportunity_value,
      ,
      drop = FALSE
    ]
    panel_title <- sprintf(
      "opportunity %s\n%s, decision t%d",
      opportunity_value,
      title_prefix,
      decision_timestep
    )
  }

  plot(
    NA,
    xlim = expand_range(value_levels, pad = 0.1),
    ylim = c(0, 1),
    xlab = x_label,
    ylab = "P(continue at current timestep)",
    main = panel_title,
    xaxt = "n",
    cex.main = 0.85
  )
  axis(1, at = value_levels)
  grid()

  if (nrow(panel_data) == 0) {
    text(mean(par("usr")[1:2]), 0.5, empty_label, cex = 0.85)
    return(invisible(NULL))
  }
  for (beta_value in beta_levels) {
    for (opp_value in opportunity_levels) {
      series_data <- panel_data[
        panel_data$beta == beta_value &
          panel_data$opportunity == opp_value,
        ,
        drop = FALSE
      ]
      if (nrow(series_data) == 0) {
        next
      }
      series_data <- series_data[order(series_data$path_value), , drop = FALSE]
      lines(
        series_data$path_value,
        series_data$p_continue,
        type = "b",
        pch = opportunity_pch[[as.character(opp_value)]],
        lty = opportunity_lty[[as.character(opp_value)]],
        col = series_color(beta_value, opp_value),
        lwd = 2
      )
    }
  }
}

plot_legacy6_value_continue_page <- function(summary_data, title_prefix, x_label, empty_label) {
  decision_timesteps <- sort(unique(suppressWarnings(as.integer(summary_data$decision_timestep))))
  decision_timesteps <- decision_timesteps[!is.na(decision_timesteps)]
  if (length(decision_timesteps) == 0) {
    decision_timesteps <- 2:min(6, tree_size)
  }

  n_cols <- max(stop_panel_count, length(decision_timesteps))
  if (identical(color_by, "opportunity")) {
    old_par <- par(mfrow = c(1, n_cols), mar = continue_panel_mar)
    for (decision_timestep in decision_timesteps) {
      plot_legacy6_value_continue_panel(
        summary_data,
        decision_timestep,
        title_prefix,
        x_label,
        empty_label
      )
    }
    blank_count <- n_cols - length(decision_timesteps)
    if (blank_count > 0) {
      for (blank_i in seq_len(blank_count)) {
        plot.new()
      }
    }
  } else {
    old_par <- par(mfrow = c(length(opportunity_levels), n_cols), mar = continue_panel_mar)
    for (opportunity_value in opportunity_levels) {
      for (decision_timestep in decision_timesteps) {
        plot_legacy6_value_continue_panel(
          summary_data,
          decision_timestep,
          title_prefix,
          x_label,
          empty_label,
          opportunity_value
        )
      }
      blank_count <- n_cols - length(decision_timesteps)
      if (blank_count > 0) {
        for (blank_i in seq_len(blank_count)) {
          plot.new()
        }
      }
    }
  }
  plot_parameter_legend()
  par(old_par)
}

disjoint_action_panel_keys <- function(summary_data, include_decision = FALSE) {
  if (include_decision && "decision_timestep" %in% names(summary_data)) {
    panel_keys <- expand.grid(
      beta = beta_levels,
      opportunity = opportunity_levels,
      decision_timestep = sort(unique(summary_data$decision_timestep)),
      KEEP.OUT.ATTRS = FALSE,
      stringsAsFactors = FALSE
    )
  } else {
    panel_keys <- expand.grid(
      beta = beta_levels,
      opportunity = opportunity_levels,
      KEEP.OUT.ATTRS = FALSE,
      stringsAsFactors = FALSE
    )
  }
  if (nrow(panel_keys) == 0) {
    panel_keys <- data.frame(
      beta = if (length(beta_levels) > 0) beta_levels[[1]] else NA_character_,
      opportunity = if (length(opportunity_levels) > 0) opportunity_levels[[1]] else NA_character_,
      stringsAsFactors = FALSE
    )
    if (include_decision) {
      panel_keys$decision_timestep <- NA_integer_
    }
  }
  panel_keys
}

plot_disjoint3x2_t1_path_action_page <- function(summary_data) {
  panel_keys <- disjoint_action_panel_keys(summary_data, FALSE)
  plot_fixed_continue_grid_panels(
    panel_keys,
    function(beta_value, opportunity_value) {
      panel_data <- summary_data[
        summary_data$beta == beta_value &
          summary_data$opportunity == opportunity_value,
        ,
        drop = FALSE
      ]
    plot_action_probability_panel(
      panel_data = panel_data,
      x_col = "reward_t1",
      y_cols = c("p_continue_current_path", "p_continue_different_path", "p_stop"),
      action_labels = c("continue current path", "continue different path", "stop"),
      action_colors = c("#2c7fb8", "#41ab5d", "#d95f0e"),
      action_ltys = c(1, 2, 1),
      action_pchs = c(16, 17, 15),
      x_label = "Reward observed at timestep 1",
      panel_title = sprintf("After t1 reward | beta %s, opportunity %s", beta_value, opportunity_value),
      empty_label = "No t2 decisions after t1 reward"
    )
    }
  )
}

plot_disjoint3x2_later_incomplete_path_action_pages <- function(summary_data) {
  decision_timesteps <- sort(unique(suppressWarnings(as.integer(summary_data$decision_timestep))))
  decision_timesteps <- decision_timesteps[!is.na(decision_timesteps)]
  if (length(decision_timesteps) == 0) {
    decision_timesteps <- 3:min(tree_size, disjoint_path_count * disjoint_nodes_per_path)
  }

  for (decision_timestep in decision_timesteps) {
    panel_keys <- disjoint_action_panel_keys(
      summary_data[summary_data$decision_timestep == decision_timestep, , drop = FALSE],
      FALSE
    )
    plot_fixed_continue_grid_panels(
      panel_keys,
      function(beta_value, opportunity_value) {
        panel_data <- summary_data[
          summary_data$decision_timestep == decision_timestep &
            summary_data$beta == beta_value &
            summary_data$opportunity == opportunity_value,
          ,
          drop = FALSE
        ]
      plot_action_probability_panel(
        panel_data = panel_data,
        x_col = "best_path_value",
        y_cols = c("p_continue_current_path", "p_continue_different_path", "p_stop"),
        action_labels = c("continue current path", "continue different path", "stop"),
        action_colors = c("#2c7fb8", "#41ab5d", "#d95f0e"),
        action_ltys = c(1, 2, 1),
        action_pchs = c(16, 17, 15),
        x_label = "Best path value observed so far",
        panel_title = sprintf(
          "Best path incomplete, decision t%d | beta %s, opportunity %s",
          decision_timestep,
          beta_value,
          opportunity_value
        ),
        empty_label = "No decisions with incomplete best path"
      )
      }
    )
  }
}

plot_disjoint3x2_later_complete_path_action_pages <- function(summary_data) {
  decision_timesteps <- sort(unique(suppressWarnings(as.integer(summary_data$decision_timestep))))
  decision_timesteps <- decision_timesteps[!is.na(decision_timesteps)]
  if (length(decision_timesteps) == 0) {
    decision_timesteps <- 3:min(tree_size, disjoint_path_count * disjoint_nodes_per_path)
  }

  for (decision_timestep in decision_timesteps) {
    panel_keys <- disjoint_action_panel_keys(
      summary_data[summary_data$decision_timestep == decision_timestep, , drop = FALSE],
      FALSE
    )
    plot_fixed_continue_grid_panels(
      panel_keys,
      function(beta_value, opportunity_value) {
        panel_data <- summary_data[
          summary_data$decision_timestep == decision_timestep &
            summary_data$beta == beta_value &
            summary_data$opportunity == opportunity_value,
          ,
          drop = FALSE
        ]
      plot_action_probability_panel(
        panel_data = panel_data,
        x_col = "best_path_value",
        y_cols = c("p_continue", "p_stop"),
        action_labels = c("continue", "stop"),
        action_colors = c("#2c7fb8", "#d95f0e"),
        action_ltys = c(1, 1),
        action_pchs = c(16, 15),
        x_label = "Best path value observed so far",
        panel_title = sprintf(
          "Best path complete, decision t%d | beta %s, opportunity %s",
          decision_timestep,
          beta_value,
          opportunity_value
        ),
        empty_label = "No decisions with complete best path"
      )
      }
    )
  }
}

plot_disjoint3x2_best_path_continue_panel <- function(
    summary_data,
    decision_timestep,
    opportunity_value = NULL) {
  value_levels <- sort(unique(suppressWarnings(as.numeric(summary_data$best_path_value))))
  value_levels <- value_levels[!is.na(value_levels)]
  if (length(value_levels) == 0) {
    value_levels <- sort(unique(suppressWarnings(as.numeric(all_data$actual_reward))))
    value_levels <- value_levels[!is.na(value_levels)]
  }
  if (length(value_levels) == 0) {
    value_levels <- c(0, 1)
  }

  if (is.null(opportunity_value)) {
    panel_data <- summary_data[
      summary_data$decision_timestep == decision_timestep,
      ,
      drop = FALSE
    ]
    panel_title <- sprintf(
      "all opportunities\nContinue at timestep %d by best path seen so far",
      decision_timestep
    )
  } else {
    panel_data <- summary_data[
      summary_data$decision_timestep == decision_timestep &
        summary_data$opportunity == opportunity_value,
      ,
      drop = FALSE
    ]
    panel_title <- sprintf(
      "opportunity %s\nContinue at timestep %d by best path seen so far",
      opportunity_value,
      decision_timestep
    )
  }

  plot(
    NA,
    xlim = expand_range(value_levels, pad = 0.1),
    ylim = c(0, 1),
    xlab = "Max observed path value so far",
    ylab = "P(continue at current timestep)",
    main = panel_title,
    xaxt = "n",
    cex.main = 0.85
  )
  axis(1, at = value_levels)
  grid()

  if (nrow(panel_data) == 0) {
    text(mean(par("usr")[1:2]), 0.5, "No decisions with observed path values", cex = 0.85)
    return(invisible(NULL))
  }

  for (beta_value in beta_levels) {
    for (opp_value in opportunity_levels) {
      series_data <- panel_data[
        panel_data$beta == beta_value &
          panel_data$opportunity == opp_value,
        ,
        drop = FALSE
      ]
      if (nrow(series_data) == 0) {
        next
      }
      series_data <- series_data[order(series_data$best_path_value), , drop = FALSE]
      lines(
        series_data$best_path_value,
        series_data$p_continue,
        type = "b",
        pch = opportunity_pch[[as.character(opp_value)]],
        lty = opportunity_lty[[as.character(opp_value)]],
        col = series_color(beta_value, opp_value),
        lwd = 2
      )
    }
  }
}

plot_disjoint3x2_best_path_continue_page <- function(summary_data) {
  decision_timesteps <- sort(unique(suppressWarnings(as.integer(summary_data$decision_timestep))))
  decision_timesteps <- decision_timesteps[!is.na(decision_timesteps)]
  if (length(decision_timesteps) == 0) {
    decision_timesteps <- 2:min(tree_size, disjoint_path_count * disjoint_nodes_per_path)
  }

  n_cols <- max(stop_panel_count, length(decision_timesteps))
  if (identical(color_by, "opportunity")) {
    old_par <- par(mfrow = c(1, n_cols), mar = continue_panel_mar)
    for (decision_timestep in decision_timesteps) {
      plot_disjoint3x2_best_path_continue_panel(summary_data, decision_timestep)
    }
    blank_count <- n_cols - length(decision_timesteps)
    if (blank_count > 0) {
      for (blank_i in seq_len(blank_count)) {
        plot.new()
      }
    }
  } else {
    old_par <- par(
      mfrow = c(length(opportunity_levels), n_cols),
      mar = continue_panel_mar
    )
    for (opportunity_value in opportunity_levels) {
      for (decision_timestep in decision_timesteps) {
        plot_disjoint3x2_best_path_continue_panel(
          summary_data,
          decision_timestep,
          opportunity_value
        )
      }
      blank_count <- n_cols - length(decision_timesteps)
      if (blank_count > 0) {
        for (blank_i in seq_len(blank_count)) {
          plot.new()
        }
      }
    }
  }
  plot_parameter_legend()
  par(old_par)
}

plot_disjoint_ever_second_node_page <- function(summary_data) {
  panel_title <- "P(ever observe second node on same path before stopping)"
  reward_levels <- sort(unique(c(
    suppressWarnings(as.numeric(summary_data$first_reward_on_path)),
    suppressWarnings(as.numeric(all_data$actual_reward))
  )))
  reward_levels <- reward_levels[!is.na(reward_levels)]
  if (length(reward_levels) == 0) {
    reward_levels <- c(0, 1)
  }

  plot(
    NA,
    xlim = expand_range(reward_levels, pad = 0.1),
    ylim = c(0, 1),
    xlab = "First reward observed on path",
    ylab = "P(ever observe second node on that path)",
    main = panel_title,
    xaxt = "n",
    cex.main = 0.9
  )
  axis(1, at = reward_levels)
  grid()

  if (nrow(summary_data) == 0) {
    text(mean(par("usr")[1:2]), 0.5, "No observed path-first rewards", cex = 0.9)
    return(invisible(NULL))
  }

  for (beta_value in beta_levels) {
    for (opportunity_value in opportunity_levels) {
      series_data <- summary_data[
        summary_data$beta == beta_value &
          summary_data$opportunity == opportunity_value,
        ,
        drop = FALSE
      ]
      if (nrow(series_data) == 0) {
        next
      }
      series_data <- series_data[order(series_data$first_reward_on_path), , drop = FALSE]
      lines(
        series_data$first_reward_on_path,
        series_data$p_ever_second_node,
        type = "b",
        pch = opportunity_pch[[as.character(opportunity_value)]],
        lty = opportunity_lty[[as.character(opportunity_value)]],
        col = series_color(beta_value, opportunity_value),
        lwd = 2
      )
    }
  }
  plot_parameter_legend()
}

stop_reward_timesteps <- continue_reward_timesteps(all_data)
if (length(stop_reward_timesteps) == 0) {
  stop_reward_timesteps <- 1
}

stop_panel_count <- length(stop_reward_timesteps) + 1
continue_heatmap_panel_count <- if (has_t3_conditioned_continue) {
  max(1, length(beta_levels) * length(opportunity_levels))
} else {
  0
}
disjoint2x2_heatmap_panel_count <- if (is_disjoint2x2) {
  max(1, length(beta_levels) * length(opportunity_levels))
} else {
  0
}
continue_heatmap_panel_count <- max(
  continue_heatmap_panel_count,
  disjoint2x2_heatmap_panel_count
)
continue_heatmap_height <- if (continue_heatmap_panel_count > 0) {
  max(5, 4 * ceiling(continue_heatmap_panel_count / min(3, continue_heatmap_panel_count)))
} else {
  5
}
continue_line_rows <- if (identical(color_by, "opportunity")) {
  1
} else {
  length(opportunity_levels)
}
plot_single_continue_line_page <- function(plot_fun, summary_data) {
  old_par <- par(
    mfrow = c(continue_line_rows, stop_panel_count),
    mar = continue_panel_mar,
    xpd = NA
  )
  plot_fun(summary_data)
  blank_count <- continue_line_rows * stop_panel_count - 1
  if (blank_count > 0) {
    for (blank_i in seq_len(blank_count)) {
      plot.new()
    }
  }
  par(old_par)
}

plot_fixed_continue_grid_panels <- function(panel_keys, plot_panel_fun) {
  panel_slots <- max(1, continue_line_rows * stop_panel_count)
  if (nrow(panel_keys) == 0) {
    panel_keys <- data.frame(
      beta = if (length(beta_levels) > 0) beta_levels[[1]] else NA_character_,
      opportunity = if (length(opportunity_levels) > 0) opportunity_levels[[1]] else NA_character_,
      stringsAsFactors = FALSE
    )
  }

  start_indices <- seq(1, nrow(panel_keys), by = panel_slots)
  for (start_i in start_indices) {
    end_i <- min(nrow(panel_keys), start_i + panel_slots - 1)
    page_keys <- panel_keys[start_i:end_i, , drop = FALSE]
    old_par <- par(
      mfrow = c(continue_line_rows, stop_panel_count),
      mar = continue_panel_mar,
      xpd = NA
    )

    for (panel_i in seq_len(nrow(page_keys))) {
      plot_panel_fun(
        page_keys$beta[[panel_i]],
        page_keys$opportunity[[panel_i]]
      )
    }

    blank_count <- panel_slots - nrow(page_keys)
    if (blank_count > 0) {
      for (blank_i in seq_len(blank_count)) {
        plot.new()
      }
    }
    par(old_par)
  }
}

if (identical(color_by, "opportunity")) {
  pdf(
    continue_pdf,
    width = max(7, 7 * stop_panel_count),
    height = max(5, continue_heatmap_height)
  )
  old_par <- par(mfrow = c(1, stop_panel_count), mar = continue_panel_mar)
  plot_initial_stop_panel()
  for (reward_timestep in stop_reward_timesteps) {
    plot_stop_panel(reward_timestep)
  }
} else {
  pdf(
    continue_pdf,
    width = max(7, 7 * stop_panel_count),
    height = max(5, 4.5 * length(opportunity_levels), continue_heatmap_height)
  )
  old_par <- par(
    mfrow = c(length(opportunity_levels), stop_panel_count),
    mar = continue_panel_mar
  )
  for (opportunity_value in opportunity_levels) {
    plot_initial_stop_panel(opportunity_value)
    for (reward_timestep in stop_reward_timesteps) {
      plot_stop_panel(reward_timestep, opportunity_value)
    }
  }
}
plot_parameter_legend()
par(old_par)
if (has_t3_conditioned_continue) {
  plot_continue_t3_conditioned_heatmaps(continue_t3_conditioned_summary)
}
if (is_bandit3) {
  plot_single_continue_line_page(
    plot_continue_t3_max_conditioned_panel,
    continue_t3_max_conditioned_summary
  )
}
if (is_bandit4) {
  plot_single_continue_line_page(
    plot_continue_t4_max_conditioned_panel,
    continue_t4_max_conditioned_summary
  )
}
if (is_disjoint2x2) {
  plot_single_continue_line_page(
    plot_disjoint2x2_t2_path_continue_panel,
    disjoint2x2_t2_path_continue_summary
  )
  plot_disjoint2x2_t3_path_continue_heatmaps(disjoint2x2_t3_path_continue_summary)
  plot_single_continue_line_page(
    plot_disjoint2x2_t3_same_path_reward_panel,
    disjoint2x2_t3_same_path_reward_summary
  )
  plot_disjoint2x2_t3_different_path_reward_heatmaps(
    disjoint2x2_t3_different_path_reward_summary
  )
  plot_disjoint2x2_t3_better_path_continue_heatmaps(
    disjoint2x2_t3_better_path_continue_summary
  )
  plot_single_continue_line_page(
    plot_disjoint2x2_t4_observed_path_reward_panel,
    disjoint2x2_t4_observed_path_reward_summary
  )
  plot_disjoint2x2_path_value_diff_continue_page(
    disjoint2x2_path_value_diff_continue_summary
  )
}
if (is_disjoint3x2) {
  plot_disjoint3x2_t1_path_action_page(disjoint3x2_t1_path_action_summary)
  plot_disjoint3x2_best_path_continue_page(
    disjoint3x2_best_path_continue_summary
  )
  plot_disjoint3x2_later_incomplete_path_action_pages(
    disjoint3x2_later_incomplete_path_action_summary
  )
  plot_disjoint3x2_later_complete_path_action_pages(
    disjoint3x2_later_complete_path_action_summary
  )
}
if (is_disjoint_tree) {
  plot_single_continue_line_page(
    plot_disjoint_ever_second_node_page,
    disjoint_ever_second_node_summary
  )
}
if (is_legacy6_tree) {
  plot_legacy6_selected_node_pages(legacy6_continue_node_summary)
  plot_legacy6_path_action_by_reward_pages(legacy6_path_action_by_reward_summary)
  plot_legacy6_value_continue_page(
    legacy6_complete_path_continue_summary,
    "Continue by fully observed path value",
    "Best fully observed path value",
    "No decisions with a fully observed path"
  )
  plot_legacy6_value_continue_page(
    legacy6_best_observed_path_continue_summary,
    "Continue by best path value seen so far",
    "Best observed path value so far",
    "No decisions with observed path values"
  )
}
dev.off()

v_mi_pdf <- file.path(
  results_dir,
  sprintf(
    "average_V_vs_MI_%s_lambda_%s_alpha_%s_beta_%s_opportunity_%s_expansion_%s_%s.pdf",
    input_type, lambda_arg, alpha_arg, beta_label, opportunity_label, expansion_label, tree_file_label
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
    "average_kl_d_%s_lambda_%s_alpha_%s_beta_%s_opportunity_%s_expansion_%s_%s.pdf",
    input_type, lambda_arg, alpha_arg, beta_label, opportunity_label, expansion_label, tree_file_label
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

kl_reward_pdf <- file.path(
  results_dir,
  sprintf(
    "kl_d_by_reward_%s_lambda_%s_alpha_%s_beta_%s_opportunity_%s_expansion_%s_%s.pdf",
    input_type, lambda_arg, alpha_arg, beta_label, opportunity_label, expansion_label, tree_file_label
  )
)

kl_reward_timesteps <- observed_reward_timesteps(all_data)
pdf(kl_reward_pdf, width = max(9.5, 4.2 * max(1, length(kl_reward_timesteps)) + 3), height = 5.5)
plot_reward_timestep_summary(
  kl_reward_summary,
  value_col = "kl_d",
  ylab = "Average kl_d",
  main_prefix = "kl_d by observed reward",
  empty_message = "No observed rewards",
  y_limits = NULL,
  timesteps = kl_reward_timesteps
)
dev.off()

reconstruction_accuracy_pdf <- file.path(
  results_dir,
  sprintf(
    "reconstruction_accuracy_by_reward_%s_lambda_%s_alpha_%s_beta_%s_opportunity_%s_expansion_%s_%s.pdf",
    input_type, lambda_arg, alpha_arg, beta_label, opportunity_label, expansion_label, tree_file_label
  )
)

pdf(
  reconstruction_accuracy_pdf,
  width = max(9.5, 4.2 * max(1, length(column_timesteps(all_data, "^estimated_reward_t[0-9]+$", "^estimated_reward_t"))) + 3),
  height = 5.5
)
plot_reward_timestep_summary(
  reconstruction_accuracy_summary,
  value_col = "accuracy",
  ylab = "Reconstruction accuracy",
  main_prefix = "Reconstruction accuracy by reward",
  empty_message = "No reconstructed rewards",
  y_limits = c(0, 1),
  timesteps = column_timesteps(all_data, "^estimated_reward_t[0-9]+$", "^estimated_reward_t")
)
dev.off()

deep_probe_accuracy_pdf <- file.path(
  results_dir,
  sprintf(
    "deep_probe_accuracy_by_reward_%s_lambda_%s_alpha_%s_beta_%s_opportunity_%s_expansion_%s_%s.pdf",
    input_type, lambda_arg, alpha_arg, beta_label, opportunity_label, expansion_label, tree_file_label
  )
)

plot_deep_probe_accuracy_panels <- function(summary_data) {
  sources <- unique(summary_data$source)
  sources <- sources[!is.na(sources)]
  decode_timesteps <- deep_probe_timesteps(all_data)
  decode_timesteps <- decode_timesteps[!is.na(decode_timesteps)]
  observed_timesteps <- sort(unique(summary_data$observed_timestep))
  observed_timesteps <- observed_timesteps[!is.na(observed_timesteps)]
  if (length(sources) == 0) {
    sources <- "none"
  }
  if (length(decode_timesteps) == 0) {
    decode_timesteps <- sort(unique(summary_data$timestep))
    decode_timesteps <- decode_timesteps[!is.na(decode_timesteps)]
  }
  if (length(decode_timesteps) == 0) decode_timesteps <- 1
  if (length(observed_timesteps) == 0) observed_timesteps <- 1

  for (source_name in sources) {
    old_par <- par(
      mfrow = c(length(observed_timesteps), length(decode_timesteps)),
      mar = c(4.5, 4.5, 2.3, 8)
    )
    for (observed_timestep in observed_timesteps) {
      for (decode_timestep in decode_timesteps) {
        panel_data <- summary_data[
          summary_data$source == source_name &
            summary_data$observed_timestep == observed_timestep &
            summary_data$timestep == decode_timestep,
          ,
          drop = FALSE
        ]

        panel_title <- sprintf("%s: decode t%d | observed t%d", source_name, decode_timestep, observed_timestep)
        if (nrow(panel_data) == 0) {
          plot(
            NA,
            xlim = c(-0.1, 1.1),
            ylim = c(0, 1),
            xlab = sprintf("Reward observed at timestep %d", observed_timestep),
            ylab = sprintf("%s probe accuracy", source_name),
            main = panel_title,
            cex.main = 0.85,
            xaxt = "n"
          )
          grid()
          empty_text <- if (decode_timestep < observed_timestep) {
            "Reward not observed yet"
          } else {
            "No held-out probe values"
          }
          text(0.5, 0.5, empty_text, cex = 0.85)
          next
        }

        plot(
          NA,
          xlim = expand_range(panel_data$reward, pad = 0.1),
          ylim = c(0, 1),
          xlab = sprintf("Reward observed at timestep %d", observed_timestep),
          ylab = sprintf("%s probe accuracy", source_name),
          main = panel_title,
          cex.main = 0.85,
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
                series_data$accuracy,
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
    }
    plot_parameter_legend()
    par(old_par)
  }
}

plot_deep_probe_t2_conditioned_heatmaps <- function(summary_data) {
  heat_cols <- grDevices::colorRampPalette(c(
    "#f7fcf0", "#e0f3db", "#ccebc5", "#a8ddb5",
    "#7bccc4", "#4eb3d3", "#2b8cbe", "#0868ac", "#084081"
  ))(64)

  sources <- unique(summary_data$source)
  sources <- sources[!is.na(sources)]
  decode_timesteps <- sort(unique(summary_data$timestep))
  decode_timesteps <- decode_timesteps[!is.na(decode_timesteps) & decode_timesteps >= 2]

  if (nrow(summary_data) == 0 || length(sources) == 0 || length(decode_timesteps) == 0) {
    old_par <- par(mfrow = c(1, 1), mar = c(4.5, 4.5, 2.5, 1))
    plot(
      NA,
      xlim = c(-0.1, 1.1),
      ylim = c(-0.1, 1.1),
      xlab = "Reward observed at timestep 1",
      ylab = "Reward observed at timestep 2",
      main = "Deep probe accuracy for reward observed at timestep 2",
      xaxt = "n",
      yaxt = "n"
    )
    text(0.5, 0.5, "No t2-conditioned probe values", cex = 0.9)
    par(old_par)
    return(invisible(NULL))
  }

  for (opportunity_value in opportunity_levels) {
    for (beta_value in beta_levels) {
      page_data <- summary_data[
        summary_data$beta == beta_value &
          summary_data$opportunity == opportunity_value,
        ,
        drop = FALSE
      ]
      if (nrow(page_data) == 0) {
        next
      }

      old_par <- par(
        mfrow = c(length(sources), length(decode_timesteps)),
        mar = c(4.6, 4.8, 2.4, 1),
        oma = c(0, 0, 3.2, 0)
      )

      for (source_name in sources) {
        for (decode_timestep in decode_timesteps) {
          panel_data <- page_data[
            page_data$source == source_name &
              page_data$timestep == decode_timestep,
            ,
            drop = FALSE
          ]

          panel_title <- sprintf("%s decode t%d", source_name, decode_timestep)
          if (nrow(panel_data) == 0) {
            plot(
              NA,
              xlim = c(-0.1, 1.1),
              ylim = c(-0.1, 1.1),
              xlab = "Reward at t1",
              ylab = "Reward at t2",
              main = panel_title,
              xaxt = "n",
              yaxt = "n"
            )
            text(0.5, 0.5, "No observations", cex = 0.85)
            next
          }

          reward_levels <- sort(unique(c(
            suppressWarnings(as.numeric(panel_data$reward_t1)),
            suppressWarnings(as.numeric(panel_data$reward_t2)),
            suppressWarnings(as.numeric(all_data$actual_reward))
          )))
          reward_levels <- reward_levels[!is.na(reward_levels)]
          if (length(reward_levels) == 0) {
            reward_levels <- sort(unique(c(panel_data$reward_t1, panel_data$reward_t2)))
          }

          z <- matrix(NA_real_, nrow = length(reward_levels), ncol = length(reward_levels))
          for (row_i in seq_len(nrow(panel_data))) {
            x_i <- match(panel_data$reward_t1[[row_i]], reward_levels)
            y_i <- match(panel_data$reward_t2[[row_i]], reward_levels)
            if (!is.na(x_i) && !is.na(y_i)) {
              z[x_i, y_i] <- panel_data$accuracy[[row_i]]
            }
          }

          image(
            reward_levels,
            reward_levels,
            z,
            zlim = c(0, 1),
            col = heat_cols,
            xlab = "Reward at t1",
            ylab = "Reward at t2",
            main = panel_title,
            xaxt = "n",
            yaxt = "n"
          )
          axis(1, at = reward_levels)
          axis(2, at = reward_levels)
          grid()
          box()

          for (row_i in seq_len(nrow(panel_data))) {
            label <- sprintf("%.2f", panel_data$accuracy[[row_i]])
            if (!is.na(panel_data$n[[row_i]])) {
              label <- sprintf("%s\nn=%d", label, panel_data$n[[row_i]])
            }
            text(
              panel_data$reward_t1[[row_i]],
              panel_data$reward_t2[[row_i]],
              labels = label,
              cex = 0.62
            )
          }
        }
      }

      mtext(
        sprintf(
          "Deep probe accuracy for reward observed at timestep 2 | beta %s, opportunity %s",
          beta_value,
          opportunity_value
        ),
        outer = TRUE,
        line = 1.1,
        font = 2
      )
      par(old_par)
    }
  }

  invisible(NULL)
}

plot_deep_probe_confusion_matrices <- function(summary_data) {
  heat_cols <- grDevices::colorRampPalette(c(
    "#f7fbff", "#deebf7", "#c6dbef", "#9ecae1",
    "#6baed6", "#4292c6", "#2171b5", "#08519c", "#08306b"
  ))(64)

  if (nrow(summary_data) == 0) {
    old_par <- par(mfrow = c(1, 1), mar = c(4.5, 4.5, 2.5, 1))
    plot(
      NA,
      xlim = c(-0.1, 1.1),
      ylim = c(-0.1, 1.1),
      xlab = "Predicted reward",
      ylab = "True reward",
      main = "Deep probe confusion matrix",
      xaxt = "n",
      yaxt = "n"
    )
    text(0.5, 0.5, "No deep probe prediction columns", cex = 0.9)
    par(old_par)
    return(invisible(NULL))
  }

  reward_levels <- sort(unique(c(
    suppressWarnings(as.numeric(summary_data$true_reward)),
    suppressWarnings(as.numeric(summary_data$pred_reward)),
    suppressWarnings(as.numeric(all_data$actual_reward))
  )))
  reward_levels <- reward_levels[!is.na(reward_levels)]
  if (length(reward_levels) == 0) {
    reward_levels <- sort(unique(c(summary_data$true_reward, summary_data$pred_reward)))
  }

  for (opportunity_value in opportunity_levels) {
    for (beta_value in beta_levels) {
      page_data_all <- summary_data[
        summary_data$beta == beta_value &
          summary_data$opportunity == opportunity_value,
        ,
        drop = FALSE
      ]
      if (nrow(page_data_all) == 0) {
        next
      }

      sources <- unique(page_data_all$source)
      sources <- sources[!is.na(sources)]
      observed_timesteps <- sort(unique(page_data_all$observed_timestep))
      observed_timesteps <- observed_timesteps[!is.na(observed_timesteps)]

      for (source_name in sources) {
        for (observed_timestep in observed_timesteps) {
          page_data <- page_data_all[
            page_data_all$source == source_name &
              page_data_all$observed_timestep == observed_timestep,
            ,
            drop = FALSE
          ]
          decode_timesteps <- sort(unique(page_data$timestep))
          decode_timesteps <- decode_timesteps[
            !is.na(decode_timesteps) &
              decode_timesteps >= observed_timestep
          ]
          if (length(decode_timesteps) == 0) {
            next
          }

          old_par <- par(
            mfrow = c(1, length(decode_timesteps)),
            mar = c(4.6, 4.8, 2.5, 1),
            oma = c(0, 0, 3.2, 0)
          )

          for (decode_timestep in decode_timesteps) {
            panel_data <- page_data[
              page_data$timestep == decode_timestep,
              ,
              drop = FALSE
            ]

            panel_title <- sprintf("probe t%d", decode_timestep)
            if (nrow(panel_data) == 0) {
              plot(
                NA,
                xlim = c(-0.1, 1.1),
                ylim = c(-0.1, 1.1),
                xlab = "Predicted reward",
                ylab = "True reward",
                main = panel_title,
                xaxt = "n",
                yaxt = "n"
              )
              text(0.5, 0.5, "No probe predictions", cex = 0.85)
              next
            }

            z <- matrix(
              NA_real_,
              nrow = length(reward_levels),
              ncol = length(reward_levels)
            )
            observed_true_rewards <- unique(panel_data$true_reward)
            observed_true_rewards <- observed_true_rewards[!is.na(observed_true_rewards)]
            for (true_reward in observed_true_rewards) {
              true_i <- match(true_reward, reward_levels)
              if (!is.na(true_i)) {
                z[, true_i] <- 0
              }
            }
            for (row_i in seq_len(nrow(panel_data))) {
              pred_i <- match(panel_data$pred_reward[[row_i]], reward_levels)
              true_i <- match(panel_data$true_reward[[row_i]], reward_levels)
              if (!is.na(pred_i) && !is.na(true_i)) {
                z[pred_i, true_i] <- panel_data$probability[[row_i]]
              }
            }

            image(
              reward_levels,
              reward_levels,
              z,
              zlim = c(0, 1),
              col = heat_cols,
              xlab = "Predicted reward",
              ylab = "True reward",
              main = panel_title,
              xaxt = "n",
              yaxt = "n"
            )
            axis(1, at = reward_levels)
            axis(2, at = reward_levels)
            grid()
            box()

            for (row_i in seq_len(nrow(panel_data))) {
              prob_value <- panel_data$probability[[row_i]]
              label <- sprintf("%.2f", prob_value)
              if (!is.na(panel_data$n[[row_i]]) && !is.na(panel_data$total_n[[row_i]])) {
                label <- sprintf("%s\n%d/%d", label, panel_data$n[[row_i]], panel_data$total_n[[row_i]])
              }
              text(
                panel_data$pred_reward[[row_i]],
                panel_data$true_reward[[row_i]],
                labels = label,
                cex = 0.58
              )
            }
          }

          mtext(
            sprintf(
              "Deep probe confusion | %s | reward observed at t%d | beta %s, opportunity %s",
              source_name,
              observed_timestep,
              beta_value,
              opportunity_value
            ),
            outer = TRUE,
            line = 1.1,
            font = 2
          )
          par(old_par)
        }
      }
    }
  }

  invisible(NULL)
}

plot_deep_probe_signed_error_panels <- function(summary_data) {
  sources <- unique(summary_data$source)
  sources <- sources[!is.na(sources)]
  decode_timesteps <- sort(unique(summary_data$timestep))
  decode_timesteps <- decode_timesteps[!is.na(decode_timesteps)]
  observed_timesteps <- sort(unique(summary_data$observed_timestep))
  observed_timesteps <- observed_timesteps[!is.na(observed_timesteps)]

  if (length(sources) == 0) {
    sources <- "none"
  }
  if (length(decode_timesteps) == 0) {
    decode_timesteps <- deep_probe_timesteps(all_data)
    decode_timesteps <- decode_timesteps[!is.na(decode_timesteps)]
  }
  if (length(decode_timesteps) == 0) decode_timesteps <- 1
  if (length(observed_timesteps) == 0) observed_timesteps <- 1

  y_values <- suppressWarnings(as.numeric(summary_data$mean_signed_error))
  y_values <- y_values[is.finite(y_values)]
  max_abs_error <- if (length(y_values) > 0) max(abs(y_values), na.rm = TRUE) else 1
  if (!is.finite(max_abs_error) || max_abs_error == 0) {
    max_abs_error <- 1
  }
  y_limits <- c(-max_abs_error, max_abs_error) * 1.08

  for (source_name in sources) {
    old_par <- par(
      mfrow = c(length(observed_timesteps), length(decode_timesteps)),
      mar = c(4.5, 4.8, 2.3, 8)
    )
    for (observed_timestep in observed_timesteps) {
      for (decode_timestep in decode_timesteps) {
        panel_data <- summary_data[
          summary_data$source == source_name &
            summary_data$observed_timestep == observed_timestep &
            summary_data$timestep == decode_timestep,
          ,
          drop = FALSE
        ]

        panel_title <- sprintf(
          "%s: signed error at t%d | observed t%d",
          source_name,
          decode_timestep,
          observed_timestep
        )
        if (nrow(panel_data) == 0) {
          plot(
            NA,
            xlim = c(-0.1, 1.1),
            ylim = y_limits,
            xlab = sprintf("Reward observed at timestep %d", observed_timestep),
            ylab = sprintf("%s signed error", source_name),
            main = panel_title,
            cex.main = 0.85,
            xaxt = "n"
          )
          abline(h = 0, lty = 2, col = "gray55")
          grid()
          empty_text <- if (decode_timestep < observed_timestep) {
            "Reward not observed yet"
          } else {
            "No held-out probe values"
          }
          text(0.5, 0, empty_text, cex = 0.85)
          next
        }

        plot(
          NA,
          xlim = expand_range(panel_data$reward, pad = 0.1),
          ylim = y_limits,
          xlab = sprintf("Reward observed at timestep %d", observed_timestep),
          ylab = sprintf("%s signed error", source_name),
          main = panel_title,
          cex.main = 0.85,
          xaxt = "n"
        )
        axis(1, at = sort(unique(panel_data$reward)))
        abline(h = 0, lty = 2, col = "gray55")
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
                series_data$mean_signed_error,
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
    }
    plot_parameter_legend()
    par(old_par)
  }

  invisible(NULL)
}

plot_deep_probe_path_context_signed_error_panels <- function(summary_data) {
  if (nrow(summary_data) == 0) {
    return(invisible(NULL))
  }

  context_order <- c(
    "same_path_other_observed",
    "other_path_one_observed",
    "other_path_both_observed"
  )
  context_labels <- c(
    same_path_other_observed = "same-path other reward observed",
    other_path_one_observed = "single observed reward on other path",
    other_path_both_observed = "other path total reward, both observed"
  )
  context_xlabels <- c(
    same_path_other_observed = "Other reward in same path",
    other_path_one_observed = "Observed reward on other path",
    other_path_both_observed = "Other path total reward"
  )

  sources <- unique(summary_data$source)
  sources <- sources[!is.na(sources)]
  contexts <- context_order[context_order %in% unique(summary_data$context_type)]
  decode_timesteps <- sort(unique(summary_data$timestep))
  decode_timesteps <- decode_timesteps[!is.na(decode_timesteps)]
  observed_timesteps <- sort(unique(summary_data$observed_timestep))
  observed_timesteps <- observed_timesteps[!is.na(observed_timesteps)]

  if (length(sources) == 0 || length(contexts) == 0) {
    return(invisible(NULL))
  }
  if (length(decode_timesteps) == 0) {
    decode_timesteps <- deep_probe_timesteps(all_data)
    decode_timesteps <- decode_timesteps[!is.na(decode_timesteps)]
  }
  if (length(decode_timesteps) == 0) decode_timesteps <- 1
  if (length(observed_timesteps) == 0) observed_timesteps <- 1

  y_values <- suppressWarnings(as.numeric(summary_data$mean_signed_error))
  y_values <- y_values[is.finite(y_values)]
  max_abs_error <- if (length(y_values) > 0) max(abs(y_values), na.rm = TRUE) else 1
  if (!is.finite(max_abs_error) || max_abs_error == 0) {
    max_abs_error <- 1
  }
  y_limits <- c(-max_abs_error, max_abs_error) * 1.08

  for (source_name in sources) {
    for (context_type in contexts) {
      old_par <- par(
        mfrow = c(length(observed_timesteps), length(decode_timesteps)),
        mar = c(4.5, 4.8, 2.7, 8)
      )

      for (observed_timestep in observed_timesteps) {
        for (decode_timestep in decode_timesteps) {
          panel_data <- summary_data[
            summary_data$source == source_name &
              summary_data$context_type == context_type &
              summary_data$observed_timestep == observed_timestep &
              summary_data$timestep == decode_timestep,
            ,
            drop = FALSE
          ]

          panel_title <- sprintf(
            "%s: signed error at t%d | target observed t%d",
            source_name,
            decode_timestep,
            observed_timestep
          )
          xlab <- context_xlabels[[context_type]]

          if (nrow(panel_data) == 0) {
            plot(
              NA,
              xlim = c(-0.1, 1.1),
              ylim = y_limits,
              xlab = xlab,
              ylab = sprintf("%s signed error", source_name),
              main = panel_title,
              cex.main = 0.82,
              xaxt = "n"
            )
            abline(h = 0, lty = 2, col = "gray55")
            grid()
            empty_text <- if (decode_timestep < observed_timestep) {
              "Target reward not observed yet"
            } else {
              "Condition not met"
            }
            text(0.5, 0, empty_text, cex = 0.85)
            next
          }

          plot(
            NA,
            xlim = expand_range(panel_data$context_reward, pad = 0.1),
            ylim = y_limits,
            xlab = xlab,
            ylab = sprintf("%s signed error", source_name),
            main = panel_title,
            cex.main = 0.82,
            xaxt = "n"
          )
          axis(1, at = sort(unique(panel_data$context_reward)))
          abline(h = 0, lty = 2, col = "gray55")
          grid()

          for (opportunity_value in opportunity_levels) {
            for (beta_value in beta_levels) {
              series_data <- panel_data[
                panel_data$beta == beta_value &
                  panel_data$opportunity == opportunity_value,
                ,
                drop = FALSE
              ]
              series_data <- series_data[order(series_data$context_reward), , drop = FALSE]
              if (nrow(series_data) > 0) {
                lines(
                  series_data$context_reward,
                  series_data$mean_signed_error,
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
      }

      mtext(
        sprintf("Path-context signed error: %s", context_labels[[context_type]]),
        outer = TRUE,
        line = 1.1,
        font = 2
      )
      plot_parameter_legend()
      par(old_par)
    }
  }

  invisible(NULL)
}

deep_probe_observed_timesteps <- sort(unique(deep_probe_accuracy_summary$observed_timestep))
deep_probe_observed_timesteps <- deep_probe_observed_timesteps[!is.na(deep_probe_observed_timesteps)]
if (length(deep_probe_observed_timesteps) == 0) {
  deep_probe_observed_timesteps <- 1
}

pdf(
  deep_probe_accuracy_pdf,
  width = max(9.5, 4.2 * max(1, length(deep_probe_timesteps(all_data))) + 3),
  height = max(5.5, 4.2 * length(deep_probe_observed_timesteps))
)
plot_deep_probe_accuracy_panels(deep_probe_accuracy_summary)
dev.off()

kl_transition_heatmap_pdf <- file.path(
  results_dir,
  sprintf(
    "kl_d_transition_heatmap_by_previous_reward_%s_lambda_%s_alpha_%s_beta_%s_opportunity_%s_expansion_%s_%s.pdf",
    input_type, lambda_arg, alpha_arg, beta_label, opportunity_label, expansion_label, tree_file_label
  )
)

blue_green_sequential_palette <- function(n = 64) {
  if (requireNamespace("ggthemes", quietly = TRUE)) {
    pal_fun <- tryCatch(
      ggthemes::tableau_color_pal("Blue-Green Sequential"),
      error = function(e) NULL
    )
    if (!is.null(pal_fun)) {
      candidate_ns <- unique(pmax(3, c(min(n, 20), 20, 12, 10, 8, 7, 6, 5, 4, 3)))
      for (candidate_n in candidate_ns) {
        cols <- tryCatch(
          pal_fun(candidate_n),
          error = function(e) NULL
        )
        if (!is.null(cols) && length(cols) > 1 && all(!is.na(cols))) {
          return(grDevices::colorRampPalette(cols)(n))
        }
      }
    }
  }

  grDevices::colorRampPalette(c(
    "#f7fcf0", "#e0f3db", "#ccebc5", "#a8ddb5",
    "#7bccc4", "#4eb3d3", "#2b8cbe", "#0868ac", "#084081"
  ))(n)
}

plot_kl_transition_heatmap_panels <- function(summary_data) {
  timesteps <- kl_transition_timesteps(all_data)
  if (length(timesteps) == 0) {
    timesteps <- 2
  }
  panel_keys <- expand.grid(
    beta = beta_levels,
    opportunity = opportunity_levels,
    timestep = timesteps,
    KEEP.OUT.ATTRS = FALSE,
    stringsAsFactors = FALSE
  )

  n_cols <- min(3, nrow(panel_keys))
  n_rows <- ceiling(nrow(panel_keys) / n_cols)
  zlim <- expand_range(summary_data$kl_d, pad = 0.05)
  heat_cols <- blue_green_sequential_palette(64)
  old_par <- par(mfrow = c(n_rows, n_cols), mar = c(4.5, 4.5, 2, 1))

  for (panel_i in seq_len(nrow(panel_keys))) {
    beta_value <- panel_keys$beta[[panel_i]]
    opportunity_value <- panel_keys$opportunity[[panel_i]]
    timestep <- panel_keys$timestep[[panel_i]]
    previous_timestep <- timestep - 1
    panel_data <- summary_data[
      summary_data$beta == beta_value &
        summary_data$opportunity == opportunity_value &
        summary_data$timestep == timestep,
      ,
      drop = FALSE
    ]

    if (nrow(panel_data) == 0) {
      plot(
        NA,
        xlim = c(-0.1, 1.1),
        ylim = c(-0.1, 1.1),
        xlab = sprintf("Reward at timestep %d", previous_timestep),
        ylab = sprintf("Reward at timestep %d", timestep),
        main = sprintf("KL t%d | beta %s, opportunity %s", timestep, beta_value, opportunity_value),
        xaxt = "n",
        yaxt = "n"
      )
      text(0.5, 0.5, sprintf("No t%d observations", timestep), cex = 0.9)
      next
    }

    x_rewards <- sort(unique(panel_data$reward_previous))
    y_rewards <- sort(unique(panel_data$reward_current))
    z <- matrix(NA_real_, nrow = length(x_rewards), ncol = length(y_rewards))
    for (row_i in seq_len(nrow(panel_data))) {
      x_i <- match(panel_data$reward_previous[[row_i]], x_rewards)
      y_i <- match(panel_data$reward_current[[row_i]], y_rewards)
      z[x_i, y_i] <- panel_data$kl_d[[row_i]]
    }

    image(
      x_rewards,
      y_rewards,
      z,
      zlim = zlim,
      col = heat_cols,
      xlab = sprintf("Reward at timestep %d", previous_timestep),
      ylab = sprintf("Reward at timestep %d", timestep),
      main = sprintf("KL t%d | beta %s, opportunity %s", timestep, beta_value, opportunity_value),
      xaxt = "n",
      yaxt = "n"
    )
    axis(1, at = x_rewards)
    axis(2, at = y_rewards)
    grid()
    for (row_i in seq_len(nrow(panel_data))) {
      text(
        panel_data$reward_previous[[row_i]],
        panel_data$reward_current[[row_i]],
        labels = format(signif(panel_data$kl_d[[row_i]], 2), trim = TRUE),
        cex = 0.7
      )
    }
  }

  par(old_par)
}

pdf(
  kl_transition_heatmap_pdf,
  width = max(8, 4 * min(3, max(1, length(beta_levels) * length(opportunity_levels) * length(kl_transition_timesteps(all_data))))),
  height = max(5, 4 * ceiling(max(1, length(beta_levels) * length(opportunity_levels) * length(kl_transition_timesteps(all_data))) / 3))
)
plot_kl_transition_heatmap_panels(kl_transition_heatmap_summary)
dev.off()

choice_stop_pdf <- file.path(
  results_dir,
  sprintf(
    "chosen_reward_given_stop_%s_lambda_%s_alpha_%s_beta_%s_opportunity_%s_expansion_%s_%s.pdf",
    input_type, lambda_arg, alpha_arg, beta_label, opportunity_label, expansion_label, tree_file_label
  )
)

choice_stop_plot_data <- choice_stop_summary
names(choice_stop_plot_data)[names(choice_stop_plot_data) == "stop_timestep"] <- "timestep"
pdf(
  choice_stop_pdf,
  width = max(9.5, 4.2 * max(1, length(column_timesteps(all_data, "^stop_t[0-9]+$", "^stop_t"))) + 3),
  height = 5.5
)
plot_reward_timestep_summary(
  choice_stop_plot_data,
  value_col = "p_choose",
  ylab = "P(choose candidate | stop)",
  main_prefix = "Choice probability by reward given stop",
  empty_message = "No stopped trials",
  y_limits = c(0, 1),
  timesteps = column_timesteps(all_data, "^stop_t[0-9]+$", "^stop_t")
)
dev.off()

weighted_choice_summary <- function(summary_data, group_cols) {
  if (nrow(summary_data) == 0) {
    return(data.frame(stringsAsFactors = FALSE))
  }

  weighted_data <- summary_data
  weighted_data$chosen_n <- weighted_data$p_choose * weighted_data$n
  count_formula <- as.formula(paste("n ~", paste(group_cols, collapse = " + ")))
  chosen_formula <- as.formula(paste("chosen_n ~", paste(group_cols, collapse = " + ")))

  choice_counts <- aggregate(count_formula, data = weighted_data, FUN = sum)
  choice_chosen <- aggregate(chosen_formula, data = weighted_data, FUN = sum)
  out <- merge(choice_counts, choice_chosen, by = group_cols)
  out$p_choose <- out$chosen_n / out$n
  out
}

plot_sequential_choice_t2 <- function(summary_data) {
  t2_data <- weighted_choice_summary(
    summary_data[summary_data$decision_timestep == 2, , drop = FALSE],
    c("beta", "opportunity", "reward_t1")
  )

  old_par <- par(mfrow = c(1, 1), mar = c(4.5, 4.5, 2.2, 8))
  if (nrow(t2_data) == 0) {
    plot(
      NA,
      xlim = c(-0.1, 1.1),
      ylim = c(0, 1),
      xlab = "Reward observed at timestep 1",
      ylab = "P(choose reward observed at timestep 2)",
      main = "Choice at timestep 2 | stopped after t2",
      xaxt = "n"
    )
    grid()
    text(0.5, 0.5, "No trials stopped after observing t1 and t2", cex = 0.9)
    par(old_par)
    return(invisible(NULL))
  }

  plot(
    NA,
    xlim = expand_range(t2_data$reward_t1, pad = 0.1),
    ylim = c(0, 1),
    xlab = "Reward observed at timestep 1",
    ylab = "P(choose reward observed at timestep 2)",
    main = "Choice at timestep 2 | stopped after t2",
    xaxt = "n"
  )
  axis(1, at = sort(unique(t2_data$reward_t1)))
  grid()

  for (opportunity_value in opportunity_levels) {
    for (beta_value in beta_levels) {
      series_data <- t2_data[
        t2_data$beta == beta_value &
          t2_data$opportunity == opportunity_value,
        ,
        drop = FALSE
      ]
      series_data <- series_data[order(series_data$reward_t1), , drop = FALSE]
      if (nrow(series_data) > 0) {
        lines(
          series_data$reward_t1,
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

  plot_parameter_legend()
  par(old_par)
}

plot_sequential_choice_t3_heatmaps <- function(summary_data) {
  t3_data <- weighted_choice_summary(
    summary_data[summary_data$decision_timestep == 3, , drop = FALSE],
    c("beta", "opportunity", "reward_t1", "reward_t2")
  )
  panel_keys <- expand.grid(
    beta = beta_levels,
    opportunity = opportunity_levels,
    KEEP.OUT.ATTRS = FALSE,
    stringsAsFactors = FALSE
  )
  n_cols <- min(3, nrow(panel_keys))
  n_rows <- ceiling(nrow(panel_keys) / n_cols)
  old_par <- par(mfrow = c(n_rows, n_cols), mar = c(4.5, 4.5, 2.2, 1))
  heat_cols <- blue_green_sequential_palette(64)

  for (panel_i in seq_len(nrow(panel_keys))) {
    beta_value <- panel_keys$beta[[panel_i]]
    opportunity_value <- panel_keys$opportunity[[panel_i]]
    panel_data <- t3_data[
      t3_data$beta == beta_value &
        t3_data$opportunity == opportunity_value,
      ,
      drop = FALSE
    ]

    if (nrow(panel_data) == 0) {
      plot(
        NA,
        xlim = c(-0.1, 1.1),
        ylim = c(-0.1, 1.1),
        xlab = "Reward observed at timestep 1",
        ylab = "Reward observed at timestep 2",
        main = sprintf("Choice at t3 | beta %s, opportunity %s", beta_value, opportunity_value),
        xaxt = "n",
        yaxt = "n"
      )
      text(0.5, 0.5, "No trials with t1, t2, and t3 observed", cex = 0.9)
      next
    }

    x_rewards <- sort(unique(panel_data$reward_t1))
    y_rewards <- sort(unique(panel_data$reward_t2))
    z <- matrix(NA_real_, nrow = length(x_rewards), ncol = length(y_rewards))
    z_n <- matrix(NA_real_, nrow = length(x_rewards), ncol = length(y_rewards))
    for (row_i in seq_len(nrow(panel_data))) {
      x_i <- match(panel_data$reward_t1[[row_i]], x_rewards)
      y_i <- match(panel_data$reward_t2[[row_i]], y_rewards)
      z[x_i, y_i] <- panel_data$p_choose[[row_i]]
      z_n[x_i, y_i] <- panel_data$n[[row_i]]
    }

    image(
      x_rewards,
      y_rewards,
      z,
      zlim = c(0, 1),
      col = heat_cols,
      xlab = "Reward observed at timestep 1",
      ylab = "Reward observed at timestep 2",
      main = sprintf("Choice at t3 | beta %s, opportunity %s", beta_value, opportunity_value),
      xaxt = "n",
      yaxt = "n"
    )
    axis(1, at = x_rewards)
    axis(2, at = y_rewards)
    grid()

    for (x_i in seq_along(x_rewards)) {
      for (y_i in seq_along(y_rewards)) {
        if (!is.na(z[x_i, y_i])) {
          text(
            x_rewards[[x_i]],
            y_rewards[[y_i]],
            sprintf("%.2f\nn=%d", z[x_i, y_i], as.integer(z_n[x_i, y_i])),
            cex = 0.65
          )
        }
      }
    }
  }

  par(old_par)
}

plot_sequential_choice_panels <- function(summary_data) {
  plot_sequential_choice_t2(summary_data)
  plot_sequential_choice_t3_heatmaps(summary_data)
}

plot_disjoint2x2_path_choice_heatmaps <- function(summary_data) {
  panel_keys <- expand.grid(
    stop_timestep = sort(unique(summary_data$stop_timestep)),
    beta = beta_levels,
    opportunity = opportunity_levels,
    KEEP.OUT.ATTRS = FALSE,
    stringsAsFactors = FALSE
  )
  if (nrow(panel_keys) == 0 || nrow(summary_data) == 0) {
    panel_keys <- data.frame(
      stop_timestep = NA_integer_,
      beta = if (length(beta_levels) > 0) beta_levels[[1]] else NA_character_,
      opportunity = if (length(opportunity_levels) > 0) opportunity_levels[[1]] else NA_character_,
      stringsAsFactors = FALSE
    )
  }

  reward_levels <- sort(unique(c(
    suppressWarnings(as.numeric(summary_data$current_path_reward)),
    suppressWarnings(as.numeric(summary_data$other_path_reward)),
    suppressWarnings(as.numeric(all_data$actual_reward))
  )))
  reward_levels <- reward_levels[!is.na(reward_levels)]
  if (length(reward_levels) == 0) {
    reward_levels <- c(0, 1)
  }

  n_cols <- min(3, max(1, nrow(panel_keys)))
  n_rows <- ceiling(nrow(panel_keys) / n_cols)
  old_par <- par(mfrow = c(n_rows, n_cols), mar = c(4.5, 4.5, 3.2, 1))
  heat_cols <- blue_green_sequential_palette(64)

  for (panel_i in seq_len(nrow(panel_keys))) {
    stop_timestep <- panel_keys$stop_timestep[[panel_i]]
    beta_value <- panel_keys$beta[[panel_i]]
    opportunity_value <- panel_keys$opportunity[[panel_i]]
    panel_data <- summary_data[
      summary_data$stop_timestep == stop_timestep &
        summary_data$beta == beta_value &
        summary_data$opportunity == opportunity_value,
      ,
      drop = FALSE
    ]

    panel_title <- sprintf(
      "P(choose candidate path) | stop t%s | beta %s, opportunity %s",
      stop_timestep,
      beta_value,
      opportunity_value
    )

    if (nrow(panel_data) == 0) {
      plot(
        NA,
        xlim = expand_range(reward_levels, pad = 0.1),
        ylim = expand_range(reward_levels, pad = 0.1),
        xlab = "Observed reward for candidate path",
        ylab = "Observed reward for other path",
        main = panel_title,
        xaxt = "n",
        yaxt = "n",
        cex.main = 0.8
      )
      axis(1, at = reward_levels)
      axis(2, at = reward_levels)
      grid()
      text(
        mean(par("usr")[1:2]),
        mean(par("usr")[3:4]),
        "No stopped trials with both paths observed",
        cex = 0.85
      )
      next
    }

    z <- matrix(NA_real_, nrow = length(reward_levels), ncol = length(reward_levels))
    z_n <- matrix(NA_real_, nrow = length(reward_levels), ncol = length(reward_levels))
    for (row_i in seq_len(nrow(panel_data))) {
      x_i <- match(panel_data$current_path_reward[[row_i]], reward_levels)
      y_i <- match(panel_data$other_path_reward[[row_i]], reward_levels)
      if (!is.na(x_i) && !is.na(y_i)) {
        z[x_i, y_i] <- panel_data$p_choose[[row_i]]
        z_n[x_i, y_i] <- panel_data$n[[row_i]]
      }
    }

    image(
      reward_levels,
      reward_levels,
      z,
      zlim = c(0, 1),
      col = heat_cols,
      xlab = "Observed reward for candidate path",
      ylab = "Observed reward for other path",
      main = panel_title,
      xaxt = "n",
      yaxt = "n",
      cex.main = 0.8
    )
    axis(1, at = reward_levels)
    axis(2, at = reward_levels)
    grid()
    for (x_i in seq_along(reward_levels)) {
      for (y_i in seq_along(reward_levels)) {
        if (!is.na(z[x_i, y_i])) {
          text(
            reward_levels[[x_i]],
            reward_levels[[y_i]],
            sprintf("%.2f\nn=%d", z[x_i, y_i], as.integer(z_n[x_i, y_i])),
            cex = 0.58
          )
        }
      }
    }
  }

  par(old_par)
}

choice_final_pdf <- file.path(
  results_dir,
  sprintf(
    "choice_probability_by_other_reward_after_observed_timestep_%s_lambda_%s_alpha_%s_beta_%s_opportunity_%s_expansion_%s_%s.pdf",
    input_type, lambda_arg, alpha_arg, beta_label, opportunity_label, expansion_label, tree_file_label
  )
)

choice_final_panel_count <- if (is_disjoint2x2) {
  max(
    1,
    length(unique(choice_final_summary$stop_timestep)) *
      length(beta_levels) *
      length(opportunity_levels)
  )
} else {
  max(1, length(beta_levels) * length(opportunity_levels))
}
choice_final_cols <- min(3, choice_final_panel_count)

pdf(
  choice_final_pdf,
  width = max(9.5, 4.2 * choice_final_cols + 1.5),
  height = max(5.5, 4.2 * ceiling(choice_final_panel_count / choice_final_cols))
)
if (is_disjoint2x2) {
  plot_disjoint2x2_path_choice_heatmaps(choice_final_summary)
} else {
  plot_sequential_choice_panels(choice_final_summary)
}
dev.off()

message("Wrote: ", continue_pdf)
message("Wrote: ", v_mi_pdf)
message("Wrote: ", kl_pdf)
message("Wrote: ", kl_reward_pdf)
message("Wrote: ", reconstruction_accuracy_pdf)
message("Wrote: ", deep_probe_accuracy_pdf)
message("Wrote: ", kl_transition_heatmap_pdf)
message("Wrote: ", choice_stop_pdf)
message("Wrote: ", choice_final_pdf)
