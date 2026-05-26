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
seeds <-7:7

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
  trial_cols <- unique(c(trial_id_cols, required_cols))
  trial_data <- unique(dat[, trial_cols, drop = FALSE])
  trial_data$reward_t1 <- suppressWarnings(as.numeric(trial_data$expanded_reward_t1))
  trial_data$reward_t2 <- suppressWarnings(as.numeric(trial_data$expanded_reward_t2))
  stop_t3_raw <- trial_data$stop_t3
  trial_data$valid_stop_t3 <- !is.na(stop_t3_raw)
  trial_data$continue_t3 <- as.numeric(!as_logical_col(stop_t3_raw))
  trial_data <- trial_data[
    !is.na(trial_data$reward_t1) &
      !is.na(trial_data$reward_t2) &
      trial_data$valid_stop_t3,
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
continue_t3_conditioned_summary <- if (is_bandit3) {
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

kl_reward_summary <- build_kl_by_reward_all_summary(all_data)
kl_transition_heatmap_summary <- build_kl_transition_heatmap_summary(all_data)
reconstruction_accuracy_summary <- build_reconstruction_accuracy_summary(all_data)
deep_probe_accuracy_summary <- build_deep_probe_accuracy_summary(all_data)
deep_probe_t2_conditioned_summary <- build_deep_probe_t2_conditioned_summary(all_data)
choice_stop_summary <- build_choice_by_stop_summary(all_data)
choice_final_summary <- build_sequential_choice_summary(all_data)

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

stop_reward_timesteps <- continue_reward_timesteps(all_data)
if (length(stop_reward_timesteps) == 0) {
  stop_reward_timesteps <- 1
}

stop_panel_count <- length(stop_reward_timesteps) + 1
continue_heatmap_panel_count <- if (is_bandit3) {
  max(1, length(beta_levels) * length(opportunity_levels))
} else {
  0
}
continue_heatmap_height <- if (is_bandit3) {
  max(5, 4 * ceiling(continue_heatmap_panel_count / min(3, continue_heatmap_panel_count)))
} else {
  5
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
if (is_bandit3) {
  plot_continue_t3_conditioned_heatmaps(continue_t3_conditioned_summary)
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
plot_deep_probe_t2_conditioned_heatmaps(deep_probe_t2_conditioned_summary)
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

choice_final_pdf <- file.path(
  results_dir,
  sprintf(
    "choice_probability_by_other_reward_after_observed_timestep_%s_lambda_%s_alpha_%s_beta_%s_opportunity_%s_expansion_%s_%s.pdf",
    input_type, lambda_arg, alpha_arg, beta_label, opportunity_label, expansion_label, tree_file_label
  )
)

pdf(
  choice_final_pdf,
  width = max(9.5, 4.2 * min(3, max(1, length(beta_levels) * length(opportunity_levels))) + 1.5),
  height = max(5.5, 4.2 * ceiling(max(1, length(beta_levels) * length(opportunity_levels)) / 3))
)
plot_sequential_choice_panels(choice_final_summary)
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
