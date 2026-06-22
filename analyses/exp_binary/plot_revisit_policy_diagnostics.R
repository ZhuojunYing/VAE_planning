#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)

get_arg <- function(i, default) {
  if (length(args) >= i && nzchar(args[[i]])) args[[i]] else default
}

beta_arg <- get_arg(1, "1000.0")
lambda_arg <- get_arg(2, "100.0")
alpha_arg <- get_arg(3, "0.0")
opportunity_arg <- get_arg(4, "0.0")
input_dir <- get_arg(5, "outputs/simulations")
results_dir <- get_arg(6, "results")
tree_size <- as.integer(get_arg(7, "3"))
input_type <- get_arg(8, "uniform")
expansion_decision_version <- get_arg(9, "lstm")
model_variant <- get_arg(10, "vae")
tree_config <- get_arg(11, "")
seed_arg <- get_arg(12, "auto")
rnn_units_arg <- get_arg(13, "64")
latent_dim_arg <- get_arg(14, "32")
simulation_source_arg <- get_arg(15, "tensorflow")
max_observations_arg <- get_arg(16, "10")
sigma_arg <- get_arg(17, "0")
optimal_dir <- get_arg(18, "analyses/exp_binary/results/bayesian_revisit_2node")
optimal_opportunity_arg <- get_arg(19, opportunity_arg)

normalize_expansion_decision_version <- function(version) {
  key <- tolower(trimws(as.character(version)))
  aliases <- c(
    "1" = "decoder", "decoder" = "decoder", "after_decoder" = "decoder",
    "2" = "lstm", "lstm" = "lstm", "after_lstm" = "lstm",
    "3" = "pre_lstm", "pre_lstm" = "pre_lstm", "before_lstm" = "pre_lstm"
  )
  if (!key %in% names(aliases)) {
    stop(sprintf("Unknown expansion_decision_version=%s.", version))
  }
  unname(aliases[[key]])
}

normalize_model_variant <- function(variant) {
  key <- tolower(trimws(as.character(variant)))
  aliases <- c("vae" = "vae", "autoencoder" = "vae", "rnn" = "rnn", "plain_rnn" = "rnn")
  if (!key %in% names(aliases)) {
    stop(sprintf("Unknown model_variant=%s.", variant))
  }
  unname(aliases[[key]])
}

normalize_tree_config <- function(config) {
  key <- tolower(trimws(as.character(config)))
  if (!nzchar(key) || key %in% c("auto", "default", "legacy")) {
    return("")
  }
  aliases <- c(
    "bandit3" = "bandit3", "3armed" = "bandit3", "3_arm" = "bandit3",
    "bandit4" = "bandit4", "4armed" = "bandit4", "4_arm" = "bandit4",
    "disjoint2x2" = "disjoint2x2", "2x2" = "disjoint2x2",
    "disjoint3x2" = "disjoint3x2", "3x2" = "disjoint3x2"
  )
  if (!key %in% names(aliases)) {
    stop(sprintf("Unknown tree_config=%s.", config))
  }
  unname(aliases[[key]])
}

normalize_simulation_source <- function(source) {
  key <- tolower(trimws(as.character(source)))
  aliases <- c("tf" = "tensorflow", "tensorflow" = "tensorflow", "keras" = "tensorflow", "jax" = "jax")
  if (!key %in% names(aliases)) {
    stop(sprintf("simulation_source must be tensorflow or jax. Got %s.", source))
  }
  unname(aliases[[key]])
}

expansion_decision_version <- normalize_expansion_decision_version(expansion_decision_version)
model_variant <- normalize_model_variant(model_variant)
tree_config <- normalize_tree_config(tree_config)
simulation_source <- normalize_simulation_source(simulation_source_arg)
if (
  identical(simulation_source, "jax") &&
    normalizePath(input_dir, mustWork = FALSE) == normalizePath("outputs/simulations", mustWork = FALSE)
) {
  input_dir <- "outputs/jax_simulations"
}

base_results_dir <- results_dir
dir.create(base_results_dir, recursive = TRUE, showWarnings = FALSE)
message(sprintf("Using %s revisit simulation CSVs from %s", simulation_source, input_dir))

tree_file_label <- paste0(tree_size, "n", if (nzchar(tree_config)) paste0("_", tree_config) else "")
architecture_file_label <- sprintf("rnn_%s_latent_%s", rnn_units_arg, latent_dim_arg)
revisit_label <- paste0("revisit_maxobs_", max_observations_arg)
source_suffix <- if (identical(simulation_source, "jax")) "_source_jax" else ""

beta_values <- trimws(strsplit(beta_arg, ",")[[1]])
opportunity_values <- trimws(strsplit(opportunity_arg, ",")[[1]])
sigma_values <- trimws(strsplit(sigma_arg, ",")[[1]])
optimal_opportunity_values <- trimws(strsplit(optimal_opportunity_arg, ",")[[1]])
auto_seeds <- identical(tolower(trimws(seed_arg)), "auto")
seed_values <- if (auto_seeds) integer() else as.integer(trimws(strsplit(seed_arg, ",")[[1]]))

arg_label <- function(values) {
  gsub("[^A-Za-z0-9._-]+", "_", paste(values, collapse = "_"))
}

short_num_label <- function(value) {
  value_num <- suppressWarnings(as.numeric(value))
  if (is.na(value_num)) {
    return(arg_label(value))
  }
  gsub("[^A-Za-z0-9._-]+", "_", format(signif(value_num, 6), scientific = FALSE, trim = TRUE))
}

compact_arg_label <- function(values, max_full_chars = 45) {
  full_label <- arg_label(values)
  if (nchar(full_label) <= max_full_chars) {
    return(full_label)
  }
  nums <- suppressWarnings(as.numeric(values))
  if (all(!is.na(nums))) {
    return(sprintf("n%d_min_%s_max_%s", length(nums), short_num_label(min(nums)), short_num_label(max(nums))))
  }
  sprintf("n%d_%s", length(values), substr(full_label, 1, max_full_chars))
}

format_plot_values <- function(values) {
  nums <- suppressWarnings(as.numeric(values))
  vapply(seq_along(values), function(i) {
    if (is.na(nums[[i]])) {
      return(as.character(values[[i]]))
    }
    format(signif(nums[[i]], 5), scientific = FALSE, trim = TRUE)
  }, character(1))
}

lambda_label <- compact_arg_label(trimws(strsplit(lambda_arg, ",")[[1]]))
alpha_label <- compact_arg_label(trimws(strsplit(alpha_arg, ",")[[1]]))
beta_label <- compact_arg_label(beta_values)
opportunity_label <- compact_arg_label(opportunity_values)
optimal_opportunity_label <- compact_arg_label(optimal_opportunity_values)
sigma_label <- compact_arg_label(sigma_values)
sigma_numeric_values <- suppressWarnings(as.numeric(sigma_values))
sigma_arg_is_default <- length(sigma_values) == 1L &&
  !is.na(sigma_numeric_values[[1]]) &&
  abs(sigma_numeric_values[[1]]) < 1e-12
sigma_file_suffix <- if (sigma_arg_is_default) "" else paste0("_sigma_", sigma_label)
same_opportunity_args <- identical(
  suppressWarnings(as.numeric(opportunity_values)),
  suppressWarnings(as.numeric(optimal_opportunity_values))
) || identical(opportunity_values, optimal_opportunity_values)
optimal_opportunity_file_suffix <- if (same_opportunity_args) {
  ""
} else {
  paste0("_optimal_opportunity_", optimal_opportunity_label)
}
file_suffix <- sprintf(
  "%s_lambda_%s_alpha_%s_beta_%s_opportunity_%s_expansion_%s_variant_%s_%s_%s_%s%s%s%s",
  input_type,
  lambda_label,
  alpha_label,
  beta_label,
  opportunity_label,
  arg_label(expansion_decision_version),
  arg_label(model_variant),
  arg_label(architecture_file_label),
  tree_file_label,
  revisit_label,
  sigma_file_suffix,
  optimal_opportunity_file_suffix,
  source_suffix
)

short_string_hash <- function(text) {
  values <- utf8ToInt(enc2utf8(as.character(text)))
  hash <- 0
  for (value in values) {
    hash <- (hash * 33 + value) %% 2147483647
  }
  format(as.integer(hash), scientific = FALSE)
}

unique_value_count <- function(values) {
  nums <- suppressWarnings(as.numeric(values))
  if (all(!is.na(nums))) {
    return(length(unique(format(signif(nums, 12), scientific = FALSE, trim = TRUE))))
  }
  length(unique(as.character(values)))
}

beta_varies <- unique_value_count(beta_values) > 1L
opportunity_varies <- unique_value_count(opportunity_values) > 1L
variation_label <- if (beta_varies && opportunity_varies) {
  "vary_beta_opportunity"
} else if (beta_varies) {
  "vary_beta"
} else if (opportunity_varies) {
  "vary_opportunity"
} else {
  "single_beta_opportunity"
}

run_folder_name <- sprintf(
  "%s_%s_%s_h%s",
  arg_label(tree_file_label),
  variation_label,
  format(Sys.time(), "%Y%m%d_%H%M%S"),
  short_string_hash(file_suffix)
)
plot_output_dir <- file.path(base_results_dir, "revisit", run_folder_name)
dir.create(plot_output_dir, recursive = TRUE, showWarnings = FALSE)
message(sprintf("Saving revisit plots to %s", plot_output_dir))

axis_filename_label <- function(value) {
  value <- gsub("_+", "_", gsub("[^A-Za-z0-9]+", "_", tolower(as.character(value))))
  value <- gsub("^_|_$", "", value)
  if (!nzchar(value)) "plot" else value
}

axis_plot_file_prefix <- function(x_col, y_col, variant = NULL) {
  prefix <- sprintf("%s_vs_%s", axis_filename_label(y_col), axis_filename_label(x_col))
  if (!is.null(variant) && nzchar(variant)) {
    prefix <- sprintf("%s_%s", prefix, axis_filename_label(variant))
  }
  prefix
}

safe_png_path <- function(file_prefix, suffix = NULL, max_basename_chars = 180) {
  basename <- sprintf("%s.png", axis_filename_label(file_prefix))
  if (nchar(basename, type = "bytes") <= max_basename_chars) {
    return(file.path(plot_output_dir, basename))
  }
  hash <- short_string_hash(basename)
  fixed_chars <- nchar(file_prefix, type = "bytes") +
    nchar(hash, type = "bytes") +
    nchar("_h.png", type = "bytes") +
    1L
  keep_chars <- max(16L, max_basename_chars - fixed_chars)
  compact_suffix <- substr(suffix, 1L, keep_chars)
  file.path(plot_output_dir, sprintf("%s_h%s.png", substr(axis_filename_label(file_prefix), 1L, keep_chars), hash))
}

panel_width_in <- 60 / 25.4
panel_height_in <- 60 / 25.4
plot_font_size_pt <- 7
legend_panel_fraction <- 0.75

open_panel_png <- function(path, n_cols = 1L, n_rows = 1L, legend_fraction = 0) {
  png(
    path,
    width = panel_width_in * (n_cols + legend_fraction),
    height = panel_height_in * n_rows,
    units = "in",
    res = 300,
    pointsize = plot_font_size_pt
  )
}

apply_panel_text_style <- function() {
  par(cex = 1, cex.axis = 1, cex.lab = 1, cex.main = 1, cex.sub = 1)
}

drop_unnamed_index_columns <- function(dat) {
  unnamed_cols <- names(dat) %in% c("", "...1", "X", "X1")
  if (any(unnamed_cols)) {
    dat <- dat[, !unnamed_cols, drop = FALSE]
  }
  dat
}

read_csv_fast <- function(path, keep_names = character(), keep_patterns = character()) {
  select_cols <- NULL
  if (requireNamespace("data.table", quietly = TRUE)) {
    header <- names(data.table::fread(path, nrows = 0, data.table = FALSE, showProgress = FALSE))
    keep <- header[header %in% keep_names]
    if (length(keep_patterns) > 0) {
      pattern <- paste(keep_patterns, collapse = "|")
      keep <- unique(c(keep, header[grepl(pattern, header)]))
    }
    if (length(keep) > 0) {
      select_cols <- keep
    }
    dat <- data.table::fread(path, select = select_cols, data.table = FALSE, showProgress = FALSE)
    return(drop_unnamed_index_columns(as.data.frame(dat)))
  }
  dat <- drop_unnamed_index_columns(read.csv(path, stringsAsFactors = FALSE))
  if (length(keep_names) == 0 && length(keep_patterns) == 0) {
    return(dat)
  }
  keep <- names(dat)[names(dat) %in% keep_names]
  if (length(keep_patterns) > 0) {
    pattern <- paste(keep_patterns, collapse = "|")
    keep <- unique(c(keep, names(dat)[grepl(pattern, names(dat))]))
  }
  dat[, keep, drop = FALSE]
}

simulation_keep_names <- c(
  "V",
  "graph",
  "chosen_path",
  "opportunity_cost",
  "observation_sigma",
  "expansion_decision_version",
  "allow_node_revisit",
  "max_observations_before_stop",
  "node",
  "actual_reward"
)
simulation_keep_patterns <- c(
  "^expanded_reward_t[0-9]+$",
  "^expanded_node_t[0-9]+$",
  "^stop_t[0-9]+$",
  "^kl_d_t[0-9]+$",
  "^action_policy_entropy_t[0-9]+$",
  "^policy_entropy_t[0-9]+$",
  "^expansion_entropy_t[0-9]+$",
  "^entropy_t[0-9]+$",
  "^action_output_path[0-9]+_t[0-9]+$",
  "^terminal_choice_prob_path[0-9]+_t[0-9]+$"
)

bayesian_keep_names <- c(
  "V",
  "graph",
  "chosen_path",
  "reward_node1",
  "reward_node2",
  "time_cost",
  "observation_sigma",
  "observations_before_stop",
  "stop_decision_timestep",
  "visits_node1",
  "visits_node2"
)
bayesian_keep_patterns <- c(
  "^expanded_reward_t[0-9]+$",
  "^expanded_node_t[0-9]+$",
  "^stop_t[0-9]+$",
  "^kl_d_t[0-9]+$",
  "^posterior_mean_[12]_t[0-9]+$",
  "^terminal_choice_prob_path[0-9]+_t[0-9]+$",
  "^action_output_path[0-9]+_t[0-9]+$"
)

value_candidates <- function(x) {
  x_chr <- as.character(x)
  x_num <- suppressWarnings(as.numeric(x_chr))
  candidates <- x_chr
  if (!is.na(x_num)) {
    candidates <- c(candidates, format(x_num, scientific = FALSE, trim = TRUE))
    if (abs(x_num - round(x_num, 1)) < 1e-12) {
      candidates <- c(candidates, sprintf("%.1f", round(x_num, 1)))
    }
    if (abs(x_num - round(x_num, 2)) < 1e-12) {
      candidates <- c(candidates, sprintf("%.2f", round(x_num, 2)))
    }
  }
  unique(candidates)
}

model_variant_file_segments <- function(variant) {
  segments <- sprintf("variant_%s_", variant)
  if (identical(variant, "vae")) {
    segments <- c(segments, "")
  }
  unique(segments)
}

simulation_tree_file_labels <- function() {
  labels <- paste0(tree_file_label, "_", architecture_file_label)
  if (identical(as.character(rnn_units_arg), "64") && identical(as.character(latent_dim_arg), "32")) {
    labels <- c(labels, tree_file_label)
  }
  unique(labels)
}

variant_file_segments <- model_variant_file_segments(model_variant)

filename_sigma_value <- function(path) {
  matches <- regexec("_obs_sigma_([^_]+)", basename(path))
  pieces <- regmatches(basename(path), matches)[[1]]
  if (length(pieces) >= 2L) {
    return(pieces[[2L]])
  }
  "0"
}

sigma_matches <- function(path, sigma_value) {
  found <- suppressWarnings(as.numeric(filename_sigma_value(path)))
  requested <- suppressWarnings(as.numeric(sigma_value))
  if (!is.na(found) && !is.na(requested)) {
    return(abs(found - requested) < 1e-8)
  }
  identical(as.character(filename_sigma_value(path)), as.character(sigma_value))
}

revisit_optional_suffix_regex <- function() {
  "(_obs_sigma_[^_]+)?"
}

numeric_file_match <- function(lambda_value, alpha_value, beta_value, opportunity_value, seed, sigma_value) {
  requested <- suppressWarnings(as.numeric(c(lambda_value, alpha_value, beta_value, opportunity_value)))
  requested_maxobs <- suppressWarnings(as.numeric(max_observations_arg))
  if (any(is.na(requested)) || is.na(requested_maxobs)) {
    return(NA_character_)
  }
  files <- list.files(input_dir, full.names = TRUE)
  for (tree_label_candidate in simulation_tree_file_labels()) {
    for (variant_file_segment in variant_file_segments) {
      pattern <- paste0(
        "^lambda_([^_]+)_alpha_([^_]+)_beta_([^_]+)_opportunity_([^_]+)_",
        "expansion_", expansion_decision_version, "_", variant_file_segment,
        "seed_", seed, "_", tree_label_candidate,
        "_revisit_maxobs_([^_]+)",
        revisit_optional_suffix_regex(),
        "_", input_type, "\\.csv$"
      )
      matches <- regexec(pattern, basename(files))
      pieces <- regmatches(basename(files), matches)
      for (i in seq_along(pieces)) {
        if (length(pieces[[i]]) == 0) {
          next
        }
        found <- suppressWarnings(as.numeric(pieces[[i]][2:5]))
        found_maxobs <- suppressWarnings(as.numeric(pieces[[i]][6]))
        if (any(is.na(found)) || is.na(found_maxobs)) {
          next
        }
        if (
          all(abs(found - requested) < 1e-8) &&
            abs(found_maxobs - requested_maxobs) < 1e-8 &&
            sigma_matches(files[[i]], sigma_value)
        ) {
          return(files[[i]])
        }
      }
    }
  }
  NA_character_
}

simulation_path <- function(lambda_value, alpha_value, beta_value, opportunity_value, seed, sigma_value) {
  for (lambda_candidate in value_candidates(lambda_value)) {
    for (alpha_candidate in value_candidates(alpha_value)) {
      for (beta_candidate in value_candidates(beta_value)) {
        for (opportunity_candidate in value_candidates(opportunity_value)) {
          for (maxobs_candidate in value_candidates(max_observations_arg)) {
            for (tree_label_candidate in simulation_tree_file_labels()) {
              for (variant_file_segment in variant_file_segments) {
                for (sigma_candidate in value_candidates(sigma_value)) {
                  sigma_num <- suppressWarnings(as.numeric(sigma_candidate))
                  suffixes <- if (!is.na(sigma_num) && abs(sigma_num) < 1e-12) {
                    c("", paste0("_obs_sigma_", sigma_candidate))
                  } else {
                    paste0("_obs_sigma_", sigma_candidate)
                  }
                  for (suffix in suffixes) {
                    file_name <- sprintf(
                      "lambda_%s_alpha_%s_beta_%s_opportunity_%s_expansion_%s_%sseed_%d_%s_revisit_maxobs_%s%s_%s.csv",
                      lambda_candidate,
                      alpha_candidate,
                      beta_candidate,
                      opportunity_candidate,
                      expansion_decision_version,
                      variant_file_segment,
                      seed,
                      tree_label_candidate,
                      maxobs_candidate,
                      suffix,
                      input_type
                    )
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
      }
    }
  }
  numeric_file_match(lambda_value, alpha_value, beta_value, opportunity_value, seed, sigma_value)
}

matching_simulation_files <- function(lambda_value, alpha_value, beta_value, opportunity_value, sigma_value) {
  requested <- suppressWarnings(as.numeric(c(lambda_value, alpha_value, beta_value, opportunity_value)))
  requested_maxobs <- suppressWarnings(as.numeric(max_observations_arg))
  if (any(is.na(requested)) || is.na(requested_maxobs)) {
    return(data.frame(path = character(), seed = integer(), stringsAsFactors = FALSE))
  }
  files <- list.files(input_dir, full.names = TRUE)
  rows <- list()
  for (tree_label_candidate in simulation_tree_file_labels()) {
    for (variant_file_segment in variant_file_segments) {
      pattern <- paste0(
        "^lambda_([^_]+)_alpha_([^_]+)_beta_([^_]+)_opportunity_([^_]+)_",
        "expansion_", expansion_decision_version, "_", variant_file_segment,
        "seed_([0-9]+)_", tree_label_candidate,
        "_revisit_maxobs_([^_]+)",
        revisit_optional_suffix_regex(),
        "_", input_type, "\\.csv$"
      )
      matches <- regexec(pattern, basename(files))
      pieces <- regmatches(basename(files), matches)
      for (i in seq_along(pieces)) {
        if (length(pieces[[i]]) == 0) {
          next
        }
        found <- suppressWarnings(as.numeric(pieces[[i]][2:5]))
        seed_value <- suppressWarnings(as.integer(pieces[[i]][6]))
        found_maxobs <- suppressWarnings(as.numeric(pieces[[i]][7]))
        if (any(is.na(found)) || is.na(seed_value) || is.na(found_maxobs)) {
          next
        }
        if (
          all(abs(found - requested) < 1e-8) &&
            abs(found_maxobs - requested_maxobs) < 1e-8 &&
            sigma_matches(files[[i]], sigma_value)
        ) {
          rows[[length(rows) + 1L]] <- data.frame(
            path = files[[i]],
            seed = seed_value,
            sigma = sigma_value,
            stringsAsFactors = FALSE
          )
        }
      }
    }
  }
  if (length(rows) == 0) {
    return(data.frame(path = character(), seed = integer(), stringsAsFactors = FALSE))
  }
  unique(do.call(rbind, rows))
}

read_seed_file <- function(beta_value, opportunity_value, seed, sigma_value) {
  file_path <- simulation_path(lambda_arg, alpha_arg, beta_value, opportunity_value, seed, sigma_value)
  if (is.na(file_path)) {
    warning(sprintf(
      "Missing revisit simulation file for beta=%s opportunity=%s sigma=%s seed=%d",
      beta_value, opportunity_value, sigma_value, seed
    ))
    return(NULL)
  }
  dat <- read_csv_fast(file_path, keep_names = simulation_keep_names, keep_patterns = simulation_keep_patterns)
  dat$beta <- beta_value
  dat$opportunity <- opportunity_value
  dat$sigma <- sigma_value
  dat$seed <- seed
  dat$model <- "VAE"
  dat$file_path <- file_path
  dat$source_file <- file_path
  dat
}

read_simulation_file <- function(file_path, beta_value, opportunity_value, seed, sigma_value) {
  dat <- read_csv_fast(file_path, keep_names = simulation_keep_names, keep_patterns = simulation_keep_patterns)
  dat$beta <- beta_value
  dat$opportunity <- opportunity_value
  dat$sigma <- sigma_value
  dat$seed <- seed
  dat$model <- "VAE"
  dat$file_path <- file_path
  dat$source_file <- file_path
  dat
}

bayes_safe_num_label <- function(value) {
  value_num <- suppressWarnings(as.numeric(value))
  if (is.na(value_num)) {
    return(gsub("[^A-Za-z0-9._-]+", "_", as.character(value)))
  }
  if (abs(value_num) < 1e-12) {
    value_num <- 0
  }
  text <- if (abs(value_num - round(value_num)) < 1e-12) {
    as.character(as.integer(round(value_num)))
  } else {
    format(signif(value_num, 12), scientific = FALSE, trim = TRUE)
  }
  gsub("\\.", "p", gsub("-", "m", text))
}

bayes_optimal_path <- function(sigma_value, opportunity_value) {
  file.path(
    optimal_dir,
    sprintf(
      "bayesian_revisit_2node_sim_sigma_%s_cost_%s_maxobs_%s.csv",
      bayes_safe_num_label(sigma_value),
      bayes_safe_num_label(opportunity_value),
      max_observations_arg
    )
  )
}

bayes_optimal_seed_path <- function(sigma_value, opportunity_value, seed_value) {
  file.path(
    optimal_dir,
    sprintf(
      "bayesian_revisit_2node_sim_sigma_%s_cost_%s_maxobs_%s_seed_%s.csv",
      bayes_safe_num_label(sigma_value),
      bayes_safe_num_label(opportunity_value),
      max_observations_arg,
      seed_value
    )
  )
}

bayes_optimal_seed_from_path <- function(path) {
  matches <- regexec("_seed_([0-9]+)\\.csv$", basename(path))
  pieces <- regmatches(basename(path), matches)[[1]]
  if (length(pieces) >= 2L) {
    return(suppressWarnings(as.integer(pieces[[2L]])))
  }
  0L
}

bayes_optimal_paths <- function(sigma_value, opportunity_value) {
  seeded_paths <- character()
  if (!auto_seeds && length(seed_values) > 0) {
    seeded_paths <- vapply(
      seed_values,
      function(seed_value) bayes_optimal_seed_path(sigma_value, opportunity_value, seed_value),
      character(1)
    )
    seeded_paths <- seeded_paths[file.exists(seeded_paths)]
  } else {
    pattern <- sprintf(
      "^bayesian_revisit_2node_sim_sigma_%s_cost_%s_maxobs_%s_seed_[0-9]+\\.csv$",
      bayes_safe_num_label(sigma_value),
      bayes_safe_num_label(opportunity_value),
      max_observations_arg
    )
    seeded_paths <- list.files(optimal_dir, pattern = pattern, full.names = TRUE)
  }
  if (length(seeded_paths) > 0) {
    return(unique(seeded_paths))
  }
  legacy_path <- bayes_optimal_path(sigma_value, opportunity_value)
  if (file.exists(legacy_path)) {
    return(legacy_path)
  }
  character()
}

terminal_choice_prob_from_means <- function(mean_1, mean_2) {
  mean_1 <- suppressWarnings(as.numeric(mean_1))
  mean_2 <- suppressWarnings(as.numeric(mean_2))
  out <- rep(NA_real_, length(mean_1))
  valid <- is.finite(mean_1) & is.finite(mean_2)
  ties <- valid & abs(mean_1 - mean_2) <= 1e-12
  out[valid & mean_1 > mean_2] <- 1
  out[valid & mean_1 < mean_2] <- 0
  out[ties] <- 0.5
  out
}

read_bayesian_optimal_file <- function(opportunity_value, sigma_value) {
  if (tree_size != 2L || nzchar(tree_config)) {
    return(NULL)
  }
  file_paths <- bayes_optimal_paths(sigma_value, opportunity_value)
  if (length(file_paths) == 0) {
    return(NULL)
  }
  pieces <- lapply(file_paths, function(file_path) {
    dat <- read_csv_fast(file_path, keep_names = bayesian_keep_names, keep_patterns = bayesian_keep_patterns)
    dat <- drop_unnamed_index_columns(dat)
    if (nrow(dat) == 0) {
      return(NULL)
    }
    dat$seed <- bayes_optimal_seed_from_path(file_path)
    dat$file_path <- file_path
    dat$source_file <- file_path
    dat
  })
  pieces <- Filter(Negate(is.null), pieces)
  if (length(pieces) == 0) {
    return(NULL)
  }
  dat <- bind_rows_fill(pieces)
  dat$beta <- if (length(beta_values) > 0) beta_values[[1]] else "optimal"
  dat$opportunity <- opportunity_value
  dat$sigma <- sigma_value
  dat$model <- "Optimal"
  dat$opportunity_cost <- suppressWarnings(as.numeric(opportunity_value))
  dat$observation_sigma <- suppressWarnings(as.numeric(sigma_value))
  dat$expansion_decision_version <- expansion_decision_version
  dat$allow_node_revisit <- TRUE
  dat$max_observations_before_stop <- suppressWarnings(as.numeric(max_observations_arg))

  posterior_t <- sort(unique(suppressWarnings(as.integer(sub(
    "^posterior_mean_1_t",
    "",
    grep("^posterior_mean_1_t[0-9]+$", names(dat), value = TRUE)
  )))))
  for (timestep in posterior_t) {
    mean_1_col <- paste0("posterior_mean_1_t", timestep)
    mean_2_col <- paste0("posterior_mean_2_t", timestep)
    prob_1 <- terminal_choice_prob_from_means(dat[[mean_1_col]], dat[[mean_2_col]])
    dat[[paste0("terminal_choice_prob_path1_t", timestep)]] <- prob_1
    dat[[paste0("terminal_choice_prob_path2_t", timestep)]] <- ifelse(is.finite(prob_1), 1 - prob_1, NA_real_)
    dat[[paste0("action_output_path1_t", timestep)]] <- dat[[paste0("terminal_choice_prob_path1_t", timestep)]]
    dat[[paste0("action_output_path2_t", timestep)]] <- dat[[paste0("terminal_choice_prob_path2_t", timestep)]]
  }

  rows <- vector("list", 2L)
  for (node_idx in 1:2) {
    node_dat <- dat
    node_dat$node <- node_idx
    node_dat$actual_reward <- suppressWarnings(as.numeric(dat[[paste0("reward_node", node_idx)]]))
    rows[[node_idx]] <- node_dat
  }
  do.call(rbind, rows)
}

bind_rows_fill <- function(data_list) {
  if (length(data_list) == 0) {
    return(NULL)
  }
  if (requireNamespace("data.table", quietly = TRUE)) {
    return(as.data.frame(data.table::rbindlist(data_list, fill = TRUE, use.names = TRUE)))
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
    for (sigma_value in sigma_values) {
      if (auto_seeds) {
        combo_files <- matching_simulation_files(lambda_arg, alpha_arg, beta_value, opportunity_value, sigma_value)
        if (nrow(combo_files) == 0) {
          warning(sprintf(
            "Missing revisit simulation files for beta=%s opportunity=%s sigma=%s",
            beta_value, opportunity_value, sigma_value
          ))
        }
        for (file_i in seq_len(nrow(combo_files))) {
          loaded_data[[length(loaded_data) + 1L]] <- read_simulation_file(
            combo_files$path[[file_i]],
            beta_value,
            opportunity_value,
            combo_files$seed[[file_i]],
            combo_files$sigma[[file_i]]
          )
        }
      } else {
        for (seed in seed_values) {
          seed_data <- read_seed_file(beta_value, opportunity_value, seed, sigma_value)
          if (!is.null(seed_data)) {
            loaded_data[[length(loaded_data) + 1L]] <- seed_data
          }
        }
      }
    }
  }
}

bayesian_optimal_supported <- tree_size == 2L && !nzchar(tree_config)
optimal_loaded_count <- 0L
if (bayesian_optimal_supported) {
  for (opportunity_value in optimal_opportunity_values) {
    for (sigma_value in sigma_values) {
      optimal_data <- read_bayesian_optimal_file(opportunity_value, sigma_value)
      if (!is.null(optimal_data) && nrow(optimal_data) > 0) {
        loaded_data[[length(loaded_data) + 1L]] <- optimal_data
        optimal_loaded_count <- optimal_loaded_count + 1L
      }
    }
  }
} else {
  message("Skipping Bayesian optimal revisit overlay; it is currently implemented only for the 2-node default task.")
}
if (optimal_loaded_count > 0L) {
  message(sprintf(
    "Loaded %d Bayesian optimal revisit simulation file(s) from %s using optimal opportunity cost(s): %s",
    optimal_loaded_count,
    optimal_dir,
    paste(optimal_opportunity_values, collapse = ",")
  ))
}

all_data <- bind_rows_fill(loaded_data)
if (is.null(all_data) || nrow(all_data) == 0) {
  stop("No revisit simulation CSVs were found. Check parameters, seeds, input_dir, and max observations.")
}

if ("opportunity_cost" %in% names(all_data)) {
  requested_opportunity <- suppressWarnings(as.numeric(unique(c(opportunity_values, optimal_opportunity_values))))
  if (all(!is.na(requested_opportunity))) {
    row_opportunity <- suppressWarnings(as.numeric(all_data$opportunity_cost))
    keep <- vapply(row_opportunity, function(x) any(abs(x - requested_opportunity) < 1e-8), logical(1))
    all_data <- all_data[keep, , drop = FALSE]
  }
}

if ("expansion_decision_version" %in% names(all_data)) {
  all_data <- all_data[
    as.character(all_data$expansion_decision_version) == expansion_decision_version,
    ,
    drop = FALSE
  ]
}

if ("allow_node_revisit" %in% names(all_data)) {
  revisit_values <- tolower(as.character(all_data$allow_node_revisit))
  keep <- is.na(all_data$allow_node_revisit) | revisit_values %in% c("true", "t", "1", "yes", "y")
  all_data <- all_data[keep, , drop = FALSE]
}

if ("max_observations_before_stop" %in% names(all_data)) {
  requested_maxobs <- suppressWarnings(as.numeric(max_observations_arg))
  row_maxobs <- suppressWarnings(as.numeric(all_data$max_observations_before_stop))
  keep <- is.na(row_maxobs) | abs(row_maxobs - requested_maxobs) < 1e-8
  all_data <- all_data[keep, , drop = FALSE]
}

if (nrow(all_data) == 0) {
  stop("No rows remained after filtering revisit metadata.")
}

if (!"model" %in% names(all_data)) {
  all_data$model <- "VAE"
}
all_data$model <- ifelse(is.na(all_data$model) | !nzchar(as.character(all_data$model)), "VAE", as.character(all_data$model))
all_data$sigma <- as.character(all_data$sigma)

sigma_value_matches <- function(values, sigma_value) {
  value_nums <- suppressWarnings(as.numeric(values))
  sigma_num <- suppressWarnings(as.numeric(sigma_value))
  if (!is.na(sigma_num) && any(!is.na(value_nums))) {
    return(!is.na(value_nums) & abs(value_nums - sigma_num) < 1e-8)
  }
  as.character(values) == as.character(sigma_value)
}

sigma_levels <- sigma_values[vapply(sigma_values, function(sigma_value) {
  any(sigma_value_matches(all_data$sigma, sigma_value))
}, logical(1))]
if (length(sigma_levels) == 0) {
  sigma_levels <- unique(as.character(all_data$sigma))
}

filter_sigma_rows <- function(dat, sigma_value) {
  if (is.null(dat) || nrow(dat) == 0 || !"sigma" %in% names(dat)) {
    return(dat)
  }
  dat[sigma_value_matches(dat$sigma, sigma_value), , drop = FALSE]
}

ordered_model_levels <- function(values) {
  values <- unique(as.character(values))
  preferred <- c("VAE", "Optimal")
  c(preferred[preferred %in% values], sort(setdiff(values, preferred)))
}

model_levels_all <- ordered_model_levels(all_data$model)

filter_model_rows <- function(dat, model_value) {
  if (is.null(dat) || nrow(dat) == 0 || !"model" %in% names(dat)) {
    return(dat)
  }
  dat[as.character(dat$model) == as.character(model_value), , drop = FALSE]
}

model_panel_title <- function(model_value, sigma_value = NULL, facet_sigma = TRUE) {
  if (isTRUE(facet_sigma) && !is.null(sigma_value)) {
    return(sprintf("%s | %s", model_value, sigma_panel_title(sigma_value)))
  }
  as.character(model_value)
}

sigma_panel_title <- function(sigma_value) {
  sprintf("sigma %s", format_plot_values(sigma_value))
}

is_bandit3 <- identical(tree_config, "bandit3") || tree_size == 3
is_bandit4 <- identical(tree_config, "bandit4")
is_disjoint2x2 <- identical(tree_config, "disjoint2x2")
is_disjoint3x2 <- identical(tree_config, "disjoint3x2")
task_path_count <- if (is_disjoint3x2) {
  3L
} else if (is_bandit4) {
  4L
} else if (is_bandit3) {
  3L
} else {
  2L
}
task_nodes_per_path <- if (is_disjoint2x2 || is_disjoint3x2) 2L else 1L
task_node_count <- task_path_count * task_nodes_per_path
reward_values <- if (identical(input_type, "binary")) c(0, 1) else c(-4, -3, -2, -1, 1, 2, 3, 4)

is_valid_task_reward <- function(x) {
  x <- suppressWarnings(as.numeric(x))
  is.finite(x) & x %in% reward_values
}

reward_difference_axis <- function(x_col) {
  signed_cols <- c("node_reward_difference", "last_visited_reward_difference")
  possible_signed <- sort(unique(as.vector(outer(reward_values, reward_values, "-"))))
  if (x_col %in% signed_cols) {
    ticks <- seq(min(possible_signed), max(possible_signed), by = 1)
    return(list(limits = range(ticks) + c(-0.5, 0.5), ticks = ticks))
  }
  if (identical(x_col, "absolute_node_reward_difference")) {
    possible_abs <- sort(unique(abs(possible_signed)))
    ticks <- seq(min(possible_abs), max(possible_abs), by = 1)
    return(list(limits = range(ticks) + c(-0.5, 0.5), ticks = ticks))
  }
  NULL
}

mean_difference_bin_width <- 0.25

bin_mean_difference <- function(x, width = mean_difference_bin_width) {
  x <- suppressWarnings(as.numeric(x))
  out <- rep(NA_real_, length(x))
  finite <- is.finite(x) & is.finite(width) & width > 0
  out[finite] <- round(x[finite] / width) * width
  out
}

path_id_for_node <- function(node) {
  node_num <- suppressWarnings(as.integer(node))
  out <- rep(NA_integer_, length(node_num))
  valid <- !is.na(node_num)
  if (is_disjoint2x2 || is_disjoint3x2) {
    out[valid] <- floor((node_num[valid] - 1L) / 2L) + 1L
  } else {
    out[valid] <- node_num[valid]
  }
  out
}

task_path_nodes <- function() {
  split(seq_len(task_node_count), rep(seq_len(task_path_count), each = task_nodes_per_path))
}

expected_best_path_reward <- function() {
  grids <- expand.grid(rep(list(reward_values), task_node_count))
  path_nodes <- task_path_nodes()
  path_rewards <- vapply(path_nodes, function(nodes) rowSums(grids[, nodes, drop = FALSE]), numeric(nrow(grids)))
  mean(apply(path_rewards, 1, max))
}

task_reward_norm <- expected_best_path_reward()

as_logical_col <- function(x) {
  if (is.logical(x)) {
    out <- x
    out[is.na(out)] <- FALSE
    return(out)
  }
  out <- tolower(as.character(x)) %in% c("true", "t", "1", "yes", "y", "stop")
  out[is.na(out)] <- FALSE
  out
}

column_timesteps <- function(dat, pattern, prefix_pattern) {
  cols <- grep(pattern, names(dat), value = TRUE)
  timesteps <- suppressWarnings(as.integer(sub(prefix_pattern, "", cols)))
  sort(timesteps[!is.na(timesteps)])
}

numeric_matrix_from_cols <- function(dat, cols) {
  if (length(cols) == 0) {
    return(matrix(numeric(), nrow = nrow(dat), ncol = 0))
  }
  as.matrix(data.frame(
    lapply(dat[, cols, drop = FALSE], function(x) suppressWarnings(as.numeric(x))),
    check.names = FALSE
  ))
}

logical_matrix_from_cols <- function(dat, cols) {
  if (length(cols) == 0) {
    return(matrix(logical(), nrow = nrow(dat), ncol = 0))
  }
  as.matrix(data.frame(
    lapply(dat[, cols, drop = FALSE], as_logical_col),
    check.names = FALSE
  ))
}

trial_key_values <- function(dat, key_cols) {
  if (length(key_cols) == 0) {
    return(rep("all", nrow(dat)))
  }
  do.call(paste, c(lapply(key_cols, function(col) as.character(dat[[col]])), sep = "\r"))
}

build_node_actual_reward_lookup <- function(dat) {
  key_cols <- intersect(c("model", "sigma", "beta", "opportunity", "seed", "graph"), names(dat))
  if (!all(c("node", "actual_reward") %in% names(dat))) {
    return(list(key_cols = key_cols, rewards = numeric(), available = FALSE))
  }
  node_cols <- unique(c(key_cols, "node", "actual_reward"))
  node_data <- dat[, node_cols, drop = FALSE]
  if (requireNamespace("data.table", quietly = TRUE)) {
    node_data <- as.data.frame(unique(data.table::as.data.table(node_data)))
  } else {
    node_data <- unique(node_data)
  }
  node_data$node <- suppressWarnings(as.integer(node_data$node))
  node_data$actual_reward <- suppressWarnings(as.numeric(node_data$actual_reward))
  node_data <- node_data[
    !is.na(node_data$node) &
      !is.na(node_data$actual_reward) &
      node_data$node >= 1L &
      node_data$node <= task_node_count,
    ,
    drop = FALSE
  ]
  if (nrow(node_data) == 0) {
    return(list(key_cols = key_cols, rewards = numeric(), available = FALSE))
  }
  node_data$key <- paste(trial_key_values(node_data, key_cols), node_data$node, sep = "\r")
  node_data <- node_data[!duplicated(node_data$key), , drop = FALSE]
  rewards <- node_data$actual_reward
  names(rewards) <- node_data$key
  list(key_cols = key_cols, rewards = rewards, available = TRUE)
}

actual_reward_for_nodes <- function(trial_data, node_values, actual_lookup) {
  out <- rep(NA_real_, length(node_values))
  if (
    is.null(actual_lookup) ||
      !isTRUE(actual_lookup$available) ||
      length(actual_lookup$rewards) == 0 ||
      !all(actual_lookup$key_cols %in% names(trial_data))
  ) {
    return(out)
  }
  node_values <- suppressWarnings(as.integer(node_values))
  valid_nodes <- !is.na(node_values) & node_values >= 1L & node_values <= task_node_count
  if (!any(valid_nodes)) {
    return(out)
  }
  keys <- paste(trial_key_values(trial_data, actual_lookup$key_cols), node_values, sep = "\r")
  matched <- actual_lookup$rewards[keys]
  out[valid_nodes] <- suppressWarnings(as.numeric(matched[valid_nodes]))
  out
}

observed_actual_reward_values <- function(trial_data, node_col, reward_col, actual_lookup) {
  node_values <- suppressWarnings(as.integer(trial_data[[node_col]]))
  actual_values <- actual_reward_for_nodes(trial_data, node_values, actual_lookup)
  fallback_values <- suppressWarnings(as.numeric(trial_data[[reward_col]]))
  ifelse(is.finite(actual_values), actual_values, fallback_values)
}

unique_trial_rows <- function(dat, required_cols) {
  trial_id_cols <- intersect(c("model", "sigma", "beta", "opportunity", "seed", "graph"), names(dat))
  trial_cols <- unique(c(trial_id_cols, required_cols))
  out <- dat[, trial_cols, drop = FALSE]
  if (requireNamespace("data.table", quietly = TRUE)) {
    return(as.data.frame(unique(data.table::as.data.table(out))))
  }
  unique(out)
}

trial_id_cols <- intersect(c("model", "sigma", "beta", "opportunity", "seed", "graph"), names(all_data))
reward_timesteps <- column_timesteps(all_data, "^expanded_reward_t[0-9]+$", "^expanded_reward_t")
node_timesteps <- column_timesteps(all_data, "^expanded_node_t[0-9]+$", "^expanded_node_t")
kl_timesteps <- column_timesteps(all_data, "^kl_d_t[0-9]+$", "^kl_d_t")
stop_timesteps <- column_timesteps(all_data, "^stop_t[0-9]+$", "^stop_t")
entropy_cols_all <- grep(
  "^(action_policy_entropy|policy_entropy|expansion_entropy|entropy)_t[0-9]+$",
  names(all_data),
  value = TRUE
)
terminal_prob_cols_all <- grep(
  "^(action_output|terminal_choice_prob)_path[0-9]+_t[0-9]+$",
  names(all_data),
  value = TRUE
)
entropy_timesteps <- sort(unique(suppressWarnings(as.integer(sub("^.*_t", "", entropy_cols_all)))))
observation_timesteps <- Reduce(intersect, list(reward_timesteps, node_timesteps))

trial_required_cols <- unique(c(
  "V",
  "chosen_path",
  paste0("expanded_node_t", node_timesteps),
  paste0("expanded_reward_t", reward_timesteps),
  paste0("stop_t", stop_timesteps),
  paste0("kl_d_t", kl_timesteps),
  entropy_cols_all,
  terminal_prob_cols_all
))
trial_required_cols <- trial_required_cols[trial_required_cols %in% names(all_data)]
trial_data <- unique_trial_rows(all_data, trial_required_cols)

reward_cols <- paste0("expanded_reward_t", reward_timesteps)
reward_cols <- reward_cols[reward_cols %in% names(trial_data)]
node_cols <- paste0("expanded_node_t", node_timesteps)
node_cols <- node_cols[node_cols %in% names(trial_data)]
kl_cols <- paste0("kl_d_t", kl_timesteps)
kl_cols <- kl_cols[kl_cols %in% names(trial_data)]
stop_cols <- paste0("stop_t", stop_timesteps)
stop_cols <- stop_cols[stop_cols %in% names(trial_data)]
entropy_col_for_timestep <- function(timestep) {
  candidates <- c(
    paste0("action_policy_entropy_t", timestep),
    paste0("policy_entropy_t", timestep),
    paste0("expansion_entropy_t", timestep),
    paste0("entropy_t", timestep)
  )
  matches <- candidates[candidates %in% names(trial_data)]
  if (length(matches) == 0) NA_character_ else matches[[1]]
}
entropy_cols <- vapply(entropy_timesteps, entropy_col_for_timestep, character(1))
entropy_cols <- entropy_cols[!is.na(entropy_cols)]
terminal_prob_cols_for_timestep <- function(timestep) {
  explicit_pattern <- paste0("^terminal_choice_prob_path[0-9]+_t", timestep, "$")
  fallback_pattern <- paste0("^action_output_path[0-9]+_t", timestep, "$")
  cols <- grep(explicit_pattern, names(trial_data), value = TRUE)
  if (length(cols) == 0L) {
    cols <- grep(fallback_pattern, names(trial_data), value = TRUE)
  }
  if (length(cols) == 0) {
    return(character())
  }
  path_idx <- suppressWarnings(as.integer(sub("^.*_path([0-9]+)_t[0-9]+$", "\\1", cols)))
  order_idx <- order(path_idx)
  cols <- cols[order_idx]
  path_idx <- path_idx[order_idx]
  cols[!duplicated(path_idx)]
}
terminal_prob_timesteps <- sort(unique(suppressWarnings(as.integer(sub("^.*_t", "", grep(
  "^(action_output|terminal_choice_prob)_path[0-9]+_t[0-9]+$",
  names(trial_data),
  value = TRUE
))))))

reward_mat <- numeric_matrix_from_cols(trial_data, reward_cols)
node_mat <- numeric_matrix_from_cols(trial_data, node_cols)
kl_mat <- numeric_matrix_from_cols(trial_data, kl_cols)
kl_mat[!is.finite(kl_mat)] <- 0
stop_mat <- logical_matrix_from_cols(trial_data, stop_cols)
stop_mat[is.na(stop_mat)] <- FALSE
entropy_mat <- numeric_matrix_from_cols(trial_data, entropy_cols)
trial_data$chosen_path_reward <- suppressWarnings(as.numeric(trial_data$V))
trial_data$normalized_chosen_path_reward <- trial_data$chosen_path_reward / task_reward_norm
trial_data$observations_before_stop <- rowSums(is.finite(reward_mat))
max_decision_timestep <- max(
  c(reward_timesteps, node_timesteps, stop_timesteps, kl_timesteps, terminal_prob_timesteps),
  na.rm = TRUE
)
if (!is.finite(max_decision_timestep)) {
  max_decision_timestep <- max(trial_data$observations_before_stop, na.rm = TRUE)
}
if (ncol(stop_mat) > 0) {
  has_explicit_stop <- rowSums(stop_mat) > 0
  first_stop_index <- max.col(stop_mat, ties.method = "first")
  first_stop_timestep <- ifelse(
    has_explicit_stop,
    stop_timesteps[first_stop_index],
    NA_real_
  )
  trial_data$stop_decision_timestep <- ifelse(
    is.finite(first_stop_timestep),
    first_stop_timestep,
    max_decision_timestep
  )
} else {
  has_explicit_stop <- rep(FALSE, nrow(trial_data))
  trial_data$stop_decision_timestep <- trial_data$observations_before_stop
}
trial_data$timestep_before_stop <- if (ncol(stop_mat) > 0) {
  pmax(suppressWarnings(as.numeric(trial_data$stop_decision_timestep)) - 1, 0)
} else {
  trial_data$observations_before_stop
}
trial_data$unique_nodes_visited <- if (ncol(node_mat) > 0) {
  visited_by_node <- sapply(seq_len(task_node_count), function(node_id) {
    rowSums(node_mat == node_id, na.rm = TRUE) > 0
  })
  if (is.null(dim(visited_by_node))) {
    as.numeric(visited_by_node)
  } else {
    rowSums(visited_by_node)
  }
} else {
  rep(0, nrow(trial_data))
}
trial_data$kl_paid_total <- rowSums(kl_mat)
if (ncol(entropy_mat) > 0) {
  entropy_finite <- is.finite(entropy_mat)
  entropy_sums <- rowSums(ifelse(entropy_finite, entropy_mat, 0))
  entropy_counts <- rowSums(entropy_finite)
  trial_data$action_policy_entropy <- ifelse(
    entropy_counts > 0,
    entropy_sums / entropy_counts,
    NA_real_
  )
} else {
  trial_data$action_policy_entropy <- NA_real_
}
trial_data$terminal_choice_entropy <- NA_real_
terminal_choice_entropy_for_timestep <- function(timestep) {
  prob_cols <- terminal_prob_cols_for_timestep(timestep)
  if (length(prob_cols) < 2L) {
    return(rep(NA_real_, nrow(trial_data)))
  }
  prob_mat <- numeric_matrix_from_cols(trial_data, prob_cols)
  prob_mat[!is.finite(prob_mat) | prob_mat <= 0] <- 0
  prob_sums <- rowSums(prob_mat)
  valid_rows <- is.finite(prob_sums) & prob_sums > 0
  entropy <- rep(NA_real_, nrow(prob_mat))
  if (any(valid_rows)) {
    normalized_prob_mat <- prob_mat[valid_rows, , drop = FALSE] / prob_sums[valid_rows]
    entropy[valid_rows] <- -rowSums(ifelse(
      normalized_prob_mat > 0,
      normalized_prob_mat * log(normalized_prob_mat),
      0
    ))
  }
  entropy
}

terminal_binary_choice_entropy_for_timestep <- function(timestep) {
  prob_cols <- terminal_prob_cols_for_timestep(timestep)
  if (length(prob_cols) < 2L) {
    return(rep(NA_real_, nrow(trial_data)))
  }
  prob_cols <- prob_cols[seq_len(2L)]
  prob_mat <- numeric_matrix_from_cols(trial_data, prob_cols)
  prob_mat[!is.finite(prob_mat) | prob_mat <= 0] <- 0
  prob_sums <- rowSums(prob_mat)
  valid_rows <- is.finite(prob_sums) & prob_sums > 0
  entropy <- rep(NA_real_, nrow(prob_mat))
  if (any(valid_rows)) {
    normalized_prob_mat <- prob_mat[valid_rows, , drop = FALSE] / prob_sums[valid_rows]
    entropy[valid_rows] <- -rowSums(ifelse(
      normalized_prob_mat > 0,
      normalized_prob_mat * log(normalized_prob_mat),
      0
    ))
  }
  entropy
}

terminal_prob_timesteps <- terminal_prob_timesteps[is.finite(terminal_prob_timesteps)]
if (length(terminal_prob_timesteps) > 0) {
  for (timestep in terminal_prob_timesteps) {
    stop_col <- paste0("stop_t", timestep)
    if (ncol(stop_mat) == 0 || !stop_col %in% names(trial_data)) {
      next
    }
    entropy_t <- terminal_choice_entropy_for_timestep(timestep)
    stopped_t <- as_logical_col(trial_data[[stop_col]])
    fill_rows <- stopped_t & !is.finite(trial_data$terminal_choice_entropy)
    trial_data$terminal_choice_entropy[fill_rows] <- entropy_t[fill_rows]
  }
  last_terminal_timestep <- max(terminal_prob_timesteps, na.rm = TRUE)
  last_entropy <- terminal_choice_entropy_for_timestep(last_terminal_timestep)
  fill_last_rows <- !has_explicit_stop & !is.finite(trial_data$terminal_choice_entropy)
  trial_data$terminal_choice_entropy[fill_last_rows] <- last_entropy[fill_last_rows]
}
keep_trial_rows <- is.finite(trial_data$normalized_chosen_path_reward)
trial_data <- trial_data[keep_trial_rows, , drop = FALSE]
reward_mat <- reward_mat[keep_trial_rows, , drop = FALSE]
node_mat <- node_mat[keep_trial_rows, , drop = FALSE]
kl_mat <- kl_mat[keep_trial_rows, , drop = FALSE]
stop_mat <- stop_mat[keep_trial_rows, , drop = FALSE]
entropy_mat <- entropy_mat[keep_trial_rows, , drop = FALSE]
has_explicit_stop <- has_explicit_stop[keep_trial_rows]

mean_or_na <- function(x) {
  x <- suppressWarnings(as.numeric(x))
  x <- x[is.finite(x)]
  if (length(x) == 0) NA_real_ else mean(x)
}

sd_or_na <- function(x) {
  x <- suppressWarnings(as.numeric(x))
  x <- x[is.finite(x)]
  if (length(x) < 2L) NA_real_ else stats::sd(x)
}

build_pre_stop_timestep_data <- function() {
  candidate_timesteps <- sort(unique(c(reward_timesteps, node_timesteps, kl_timesteps)))
  candidate_timesteps <- candidate_timesteps[is.finite(candidate_timesteps) & candidate_timesteps > 0]
  if (length(candidate_timesteps) == 0) {
    return(data.frame())
  }
  rows <- list()
  for (observation_timestep in candidate_timesteps) {
    before_stop <- is.finite(suppressWarnings(as.numeric(trial_data$timestep_before_stop))) &
      suppressWarnings(as.numeric(trial_data$timestep_before_stop)) >= observation_timestep
    if (!any(before_stop)) {
      next
    }
    decision_timestep <- observation_timestep + 1L
    kl_col <- paste0("kl_d_t", observation_timestep)
    kl_values <- if (kl_col %in% names(trial_data)) {
      suppressWarnings(as.numeric(trial_data[[kl_col]]))
    } else {
      rep(NA_real_, nrow(trial_data))
    }
    entropy_values <- terminal_binary_choice_entropy_for_timestep(decision_timestep)
    rows[[length(rows) + 1L]] <- data.frame(
      model = trial_data$model[before_stop],
      sigma = trial_data$sigma[before_stop],
      beta = trial_data$beta[before_stop],
      opportunity = trial_data$opportunity[before_stop],
      seed = trial_data$seed[before_stop],
      timestep = observation_timestep,
      decision_timestep = decision_timestep,
      kl_paid_at_timestep = kl_values[before_stop],
      terminal_binary_choice_entropy_at_timestep = entropy_values[before_stop],
      stringsAsFactors = FALSE
    )
  }
  if (length(rows) == 0) {
    return(data.frame())
  }
  do.call(rbind, rows)
}

aggregate_means_by <- function(dat, group_cols, value_cols) {
  group_cols <- group_cols[group_cols %in% names(dat)]
  value_cols <- value_cols[value_cols %in% names(dat)]
  if (length(value_cols) == 0) {
    return(unique(dat[, group_cols, drop = FALSE]))
  }
  if (requireNamespace("data.table", quietly = TRUE)) {
    dt <- data.table::as.data.table(dat)
    out <- dt[, lapply(.SD, mean_or_na), by = group_cols, .SDcols = value_cols]
    return(as.data.frame(out))
  }
  pieces <- lapply(value_cols, function(value_col) {
    out <- aggregate(dat[[value_col]], by = dat[, group_cols, drop = FALSE], FUN = mean_or_na)
    names(out)[names(out) == "x"] <- value_col
    out
  })
  Reduce(function(left, right) merge(left, right, by = group_cols, all = TRUE), pieces)
}

add_seed_sd_by <- function(summary_data, dat, group_cols, value_cols) {
  group_cols <- group_cols[group_cols %in% names(dat)]
  value_cols <- value_cols[value_cols %in% names(dat)]
  if (nrow(summary_data) == 0 || length(value_cols) == 0 || !"seed" %in% names(dat)) {
    return(summary_data)
  }
  seed_group_cols <- unique(c(group_cols, "seed"))
  seed_means <- aggregate_means_by(dat, seed_group_cols, value_cols)
  if (nrow(seed_means) == 0) {
    return(summary_data)
  }
  if (requireNamespace("data.table", quietly = TRUE)) {
    dt <- data.table::as.data.table(seed_means)
    sd_data <- dt[, lapply(.SD, sd_or_na), by = group_cols, .SDcols = value_cols]
    sd_data <- as.data.frame(sd_data)
  } else {
    pieces <- lapply(value_cols, function(value_col) {
      out <- aggregate(seed_means[[value_col]], by = seed_means[, group_cols, drop = FALSE], FUN = sd_or_na)
      names(out)[names(out) == "x"] <- value_col
      out
    })
    sd_data <- Reduce(function(left, right) merge(left, right, by = group_cols, all = TRUE), pieces)
  }
  names(sd_data)[names(sd_data) %in% value_cols] <- paste0(names(sd_data)[names(sd_data) %in% value_cols], "_seed_sd")
  merge(summary_data, sd_data, by = group_cols, all.x = TRUE)
}

count_rows_by <- function(dat, group_cols, count_name = "n") {
  group_cols <- group_cols[group_cols %in% names(dat)]
  if (length(group_cols) == 0) {
    out <- data.frame(n = nrow(dat))
    names(out) <- count_name
    return(out)
  }
  if (requireNamespace("data.table", quietly = TRUE)) {
    dt <- data.table::as.data.table(dat)
    out <- dt[, .N, by = group_cols]
    data.table::setnames(out, "N", count_name)
    return(as.data.frame(out))
  }
  out <- aggregate(
    list(.count = rep(1, nrow(dat))),
    by = dat[, group_cols, drop = FALSE],
    FUN = sum
  )
  names(out)[names(out) == ".count"] <- count_name
  out
}

pre_stop_timestep_data <- build_pre_stop_timestep_data()
if (nrow(pre_stop_timestep_data) > 0) {
  pre_stop_timestep_summary <- aggregate_means_by(
    pre_stop_timestep_data,
    group_cols = c("model", "sigma", "beta", "opportunity", "timestep"),
    value_cols = c(
      "kl_paid_at_timestep",
      "terminal_binary_choice_entropy_at_timestep"
    )
  )
  pre_stop_timestep_count <- count_rows_by(
    pre_stop_timestep_data,
    c("model", "sigma", "beta", "opportunity", "timestep")
  )
  pre_stop_timestep_summary <- merge(
    pre_stop_timestep_summary,
    pre_stop_timestep_count,
    by = c("model", "sigma", "beta", "opportunity", "timestep"),
    all.x = TRUE
  )
} else {
  pre_stop_timestep_summary <- data.frame()
}
if (nrow(pre_stop_timestep_data) > 0) {
  pre_stop_entropy_combo_summary <- aggregate_means_by(
    pre_stop_timestep_data,
    group_cols = c("model", "sigma", "beta", "opportunity"),
    value_cols = c("terminal_binary_choice_entropy_at_timestep")
  )
  names(pre_stop_entropy_combo_summary)[
    names(pre_stop_entropy_combo_summary) == "terminal_binary_choice_entropy_at_timestep"
  ] <- "terminal_choice_entropy_combined_reached"
} else {
  pre_stop_entropy_combo_summary <- data.frame()
}

average_summary <- aggregate_means_by(
  trial_data,
  group_cols = c("model", "sigma", "beta", "opportunity"),
  value_cols = c(
    "normalized_chosen_path_reward",
    "observations_before_stop",
    "stop_decision_timestep",
    "timestep_before_stop",
    "unique_nodes_visited",
    "kl_paid_total",
    "terminal_choice_entropy"
  )
)
count_data <- count_rows_by(trial_data, c("model", "sigma", "beta", "opportunity"))
average_summary <- merge(average_summary, count_data, by = c("model", "sigma", "beta", "opportunity"), all.x = TRUE)
if (nrow(pre_stop_entropy_combo_summary) > 0) {
  average_summary <- merge(
    average_summary,
    pre_stop_entropy_combo_summary,
    by = c("model", "sigma", "beta", "opportunity"),
    all.x = TRUE
  )
}

parameter_value_matches <- function(values, target, tol = 1e-8) {
  value_nums <- suppressWarnings(as.numeric(values))
  target_num <- suppressWarnings(as.numeric(target))
  if (!is.na(target_num) && any(!is.na(value_nums))) {
    return(!is.na(value_nums) & abs(value_nums - target_num) < tol)
  }
  as.character(values) == as.character(target)
}

requested_levels_present <- function(requested, observed) {
  requested[vapply(
    requested,
    function(value) any(parameter_value_matches(observed, value)),
    logical(1)
  )]
}

lookup_named_parameter <- function(named_values, key, default = NA_character_) {
  if (length(named_values) == 0) {
    return(default)
  }
  direct <- unname(named_values[as.character(key)])
  if (length(direct) > 0 && !is.na(direct)) {
    return(direct)
  }
  key_num <- suppressWarnings(as.numeric(key))
  name_nums <- suppressWarnings(as.numeric(names(named_values)))
  if (!is.na(key_num) && any(!is.na(name_nums))) {
    hit <- which(!is.na(name_nums) & abs(name_nums - key_num) < 1e-8)
    if (length(hit) > 0) {
      return(unname(named_values[[hit[[1]]]]))
    }
  }
  default
}

beta_levels <- requested_levels_present(beta_values, all_data$beta)
if (length(beta_levels) == 0) {
  beta_levels <- unique(all_data$beta)
}
model_values_for_levels <- if ("model" %in% names(all_data)) as.character(all_data$model) else rep("VAE", nrow(all_data))
vae_level_data <- all_data[model_values_for_levels != "Optimal", , drop = FALSE]
optimal_level_data <- all_data[model_values_for_levels == "Optimal", , drop = FALSE]
vae_opportunity_levels <- requested_levels_present(opportunity_values, vae_level_data$opportunity)
if (length(vae_opportunity_levels) == 0) {
  vae_opportunity_levels <- unique(vae_level_data$opportunity)
}
optimal_opportunity_levels <- requested_levels_present(optimal_opportunity_values, optimal_level_data$opportunity)
if (length(optimal_opportunity_levels) == 0) {
  optimal_opportunity_levels <- unique(optimal_level_data$opportunity)
}
opportunity_levels <- unique(c(vae_opportunity_levels, optimal_opportunity_levels))
if (length(opportunity_levels) == 0) {
  opportunity_levels <- unique(all_data$opportunity)
}

color_luminance <- function(cols) {
  rgb_cols <- grDevices::col2rgb(cols) / 255
  0.2126 * rgb_cols[1, ] + 0.7152 * rgb_cols[2, ] + 0.0722 * rgb_cols[3, ]
}

numeric_gradient_colors <- function(levels, palette, darker_high = TRUE) {
  if (length(levels) == 0) {
    return(setNames(character(), character()))
  }
  cols <- grDevices::hcl.colors(max(3, length(levels) + 2), palette = palette)
  cols <- cols[seq_along(levels)]
  light_to_dark <- cols[order(color_luminance(cols), decreasing = TRUE)]
  nums <- suppressWarnings(as.numeric(levels))
  level_order <- if (all(!is.na(nums))) order(nums) else order(as.character(levels))
  assigned <- character(length(levels))
  assigned[level_order] <- if (darker_high) light_to_dark else rev(light_to_dark)
  setNames(assigned, levels)
}

color_by <- if (length(vae_opportunity_levels) > 1 && length(beta_levels) == 1) "opportunity" else "beta"
color_levels <- if (identical(color_by, "opportunity")) vae_opportunity_levels else beta_levels
color_cols <- if (identical(color_by, "opportunity")) {
  numeric_gradient_colors(color_levels, "Blues", darker_high = TRUE)
} else {
  numeric_gradient_colors(color_levels, "Greens", darker_high = FALSE)
}
optimal_opportunity_cols <- numeric_gradient_colors(optimal_opportunity_levels, "Purples", darker_high = TRUE)

series_color <- function(beta_value, opportunity_value) {
  if (identical(color_by, "opportunity")) {
    lookup_named_parameter(color_cols, opportunity_value)
  } else {
    lookup_named_parameter(color_cols, beta_value)
  }
}

optimal_opportunity_color <- function(opportunity_value) {
  cols <- vapply(
    opportunity_value,
    function(value) lookup_named_parameter(optimal_opportunity_cols, value, default = NA_character_),
    character(1)
  )
  missing_cols <- is.na(cols)
  if (any(missing_cols)) {
    cols[missing_cols] <- "purple4"
  }
  cols
}

model_pch_for <- function(model) {
  ifelse(as.character(model) == "Optimal", 24, 19)
}

model_point_col_for <- function(model, fill_cols) {
  model <- as.character(model)
  ifelse(model == "Optimal", "black", fill_cols)
}

model_point_bg_for <- function(model, fill_cols) {
  model <- as.character(model)
  ifelse(model == "Optimal", fill_cols, fill_cols)
}

draw_model_points <- function(x, y, model, fill_cols, cex = 1.35) {
  if (length(x) == 0) {
    return(invisible(NULL))
  }
  if (is.null(model)) {
    points(x, y, pch = 19, cex = cex, col = fill_cols)
    return(invisible(NULL))
  }
  points(
    x,
    y,
    pch = model_pch_for(model),
    cex = cex,
    col = model_point_col_for(model, fill_cols),
    bg = model_point_bg_for(model, fill_cols),
    lwd = ifelse(as.character(model) == "Optimal", 0.65, 1)
  )
  invisible(NULL)
}

point_color_for <- function(beta, opportunity, model = NULL, alpha = 0.55) {
  cols <- mapply(series_color, beta, opportunity)
  if (!is.null(model)) {
    model <- as.character(model)
    optimal_rows <- model == "Optimal"
    cols[optimal_rows] <- optimal_opportunity_color(opportunity[optimal_rows])
  }
  grDevices::adjustcolor(cols, alpha.f = alpha)
}

line_color_for <- function(beta, opportunity, model = NULL) {
  cols <- mapply(series_color, beta, opportunity)
  if (!is.null(model)) {
    model <- as.character(model)
    optimal_rows <- model == "Optimal"
    cols[optimal_rows] <- optimal_opportunity_color(opportunity[optimal_rows])
  }
  cols
}

expand_range <- function(x, pad = 0.05, default = c(0, 1)) {
  x <- suppressWarnings(as.numeric(x))
  x <- x[is.finite(x)]
  if (length(x) == 0) {
    return(default)
  }
  rng <- range(x)
  if (abs(diff(rng)) < 1e-12) {
    return(rng + c(-1, 1))
  }
  rng + c(-1, 1) * diff(rng) * pad
}

expand_log_range <- function(x, pad = 0.05, default = c(1e-3, 1)) {
  x <- suppressWarnings(as.numeric(x))
  x <- x[is.finite(x) & x > 0]
  if (length(x) == 0) {
    return(default)
  }
  log_rng <- range(log10(x))
  if (abs(diff(log_rng)) < 1e-12) {
    center <- log_rng[[1]]
    return(10 ^ (center + c(-0.5, 0.5)))
  }
  10 ^ (log_rng + c(-1, 1) * diff(log_rng) * pad)
}

plot_parameter_legend <- function(active_models = model_levels_all) {
  plot.new()
  legend_title <- if (identical(color_by, "opportunity")) "opportunity" else "beta"
  legend_labels <- paste("VAE", legend_title, format_plot_values(color_levels))
  legend_cols <- unname(color_cols[color_levels])
  legend_bg <- legend_cols
  legend_pch <- rep(19, length(color_levels))
  legend_lty <- rep(1, length(color_levels))
  if ("Optimal" %in% active_models && length(optimal_opportunity_levels) > 0) {
    optimal_cols <- unname(optimal_opportunity_cols[optimal_opportunity_levels])
    legend_labels <- c(
      legend_labels,
      paste("optimal opportunity", format_plot_values(optimal_opportunity_levels))
    )
    legend_cols <- c(legend_cols, rep("black", length(optimal_opportunity_levels)))
    legend_bg <- c(legend_bg, optimal_cols)
    legend_pch <- c(legend_pch, rep(24, length(optimal_opportunity_levels)))
    legend_lty <- c(legend_lty, rep(1, length(optimal_opportunity_levels)))
  }
  legend(
    "center",
    legend = legend_labels,
    col = legend_cols,
    pt.bg = legend_bg,
    pch = legend_pch,
    lty = legend_lty,
    bty = "n",
    cex = 0.78
  )
}

sigma_cols <- numeric_gradient_colors(sigma_levels, "Oranges", darker_high = FALSE)

sigma_color_for <- function(sigma, alpha = 0.65) {
  cols <- unname(sigma_cols[as.character(sigma)])
  missing_cols <- is.na(cols)
  if (any(missing_cols)) {
    cols[missing_cols] <- "darkorange"
  }
  grDevices::adjustcolor(cols, alpha.f = alpha)
}

plot_sigma_legend <- function(active_models = model_levels_all) {
  plot.new()
  legend_labels <- paste("sigma", format_plot_values(sigma_levels))
  legend_cols <- unname(sigma_cols[sigma_levels])
  legend_bg <- legend_cols
  legend_pch <- rep(19, length(sigma_levels))
  legend_lty <- rep(1, length(sigma_levels))
  if ("Optimal" %in% active_models) {
    legend_labels <- c(legend_labels, "VAE dot", "optimal triangle")
    legend_cols <- c(legend_cols, "gray30", "gray30")
    legend_bg <- c(legend_bg, "gray30", "gray70")
    legend_pch <- c(legend_pch, 19, 24)
    legend_lty <- c(legend_lty, NA, NA)
  }
  legend(
    "center",
    legend = legend_labels,
    col = legend_cols,
    pt.bg = legend_bg,
    pch = legend_pch,
    lty = legend_lty,
    bty = "n",
    cex = 0.8
  )
}

log_arg_for_axes <- function(log_x = FALSE, log_y = FALSE) {
  paste0(if (isTRUE(log_x)) "x" else "", if (isTRUE(log_y)) "y" else "")
}

filter_plot_data <- function(dat, x_col, y_col, log_x = FALSE, log_y = FALSE, file_prefix = "plot") {
  if (is.null(dat) || nrow(dat) == 0 || !all(c(x_col, y_col) %in% names(dat))) {
    warning(sprintf("Missing data or columns for %s; skipping.", file_prefix))
    return(data.frame())
  }
  x <- suppressWarnings(as.numeric(dat[[x_col]]))
  y <- suppressWarnings(as.numeric(dat[[y_col]]))
  keep <- is.finite(x) & is.finite(y)
  if (isTRUE(log_x)) {
    keep <- keep & x > 0
  }
  if (isTRUE(log_y)) {
    keep <- keep & y > 0
  }
  out <- dat[keep, , drop = FALSE]
  if (nrow(out) == 0) {
    warning(sprintf("No finite%s%s points for %s; skipping.",
      if (isTRUE(log_x)) " positive-x" else "",
      if (isTRUE(log_y)) " positive-y" else "",
      file_prefix
    ))
  }
  out
}

plot_summary_scatter <- function(
  summary_data,
  x_col,
  y_col,
  xlab,
  ylab,
  file_prefix,
  log_x = FALSE,
  log_y = FALSE,
  facet_sigma = TRUE,
  color_mode = c("parameter", "sigma"),
  model_layout = c("rows", "overlay", "vae_only", "rows_present")
) {
  color_mode <- match.arg(color_mode)
  model_layout <- match.arg(model_layout)
  path <- safe_png_path(file_prefix, file_suffix)
  if (identical(model_layout, "vae_only") && "model" %in% names(summary_data)) {
    summary_data <- filter_model_rows(summary_data, "VAE")
  }
  plot_data <- filter_plot_data(summary_data, x_col, y_col, log_x, log_y, file_prefix)
  if (nrow(plot_data) == 0) {
    return(invisible(NULL))
  }
  panel_levels <- if (isTRUE(facet_sigma)) sigma_levels else "all"
  n_panels <- max(1L, length(panel_levels))
  model_levels <- if (identical(model_layout, "overlay")) {
    "all"
  } else if (identical(model_layout, "rows_present") && "model" %in% names(plot_data)) {
    ordered_model_levels(plot_data$model)
  } else if ("model" %in% names(summary_data)) {
    ordered_model_levels(summary_data$model)
  } else {
    "VAE"
  }
  legend_model_levels <- if (identical(model_layout, "overlay") && "model" %in% names(summary_data)) {
    ordered_model_levels(summary_data$model)
  } else if (identical(model_layout, "rows_present") && "model" %in% names(plot_data)) {
    ordered_model_levels(plot_data$model)
  } else {
    model_levels
  }
  n_models <- max(1L, length(model_levels))
  open_panel_png(path, n_cols = n_panels, n_rows = n_models, legend_fraction = legend_panel_fraction)
  old_par <- par(no.readonly = TRUE)
  panel_matrix <- matrix(seq_len(n_panels * n_models), nrow = n_models, ncol = n_panels, byrow = TRUE)
  layout(cbind(panel_matrix, rep(n_panels * n_models + 1L, n_models)),
    widths = c(rep(1, n_panels), legend_panel_fraction)
  )
  par(mar = c(4.2, 4.2, if (isTRUE(facet_sigma) && n_panels > 1L) 2 else 1, 1))
  apply_panel_text_style()
  x_limits <- if (isTRUE(log_x)) {
    expand_log_range(plot_data[[x_col]], pad = 0.05)
  } else {
    expand_range(plot_data[[x_col]], pad = 0.05)
  }
  y_limits <- if (isTRUE(log_y)) {
    expand_log_range(plot_data[[y_col]], pad = 0.05)
  } else {
    expand_range(plot_data[[y_col]], pad = 0.05)
  }
  for (model_value in model_levels) {
    model_data <- if (!identical(model_value, "all") && "model" %in% names(plot_data)) {
      filter_model_rows(plot_data, model_value)
    } else {
      plot_data
    }
    for (panel_value in panel_levels) {
      piece <- if (isTRUE(facet_sigma)) filter_sigma_rows(model_data, panel_value) else model_data
      point_cols <- character()
      if (nrow(piece) > 0) {
        point_cols <- if (identical(color_mode, "sigma")) {
          sigma_color_for(piece$sigma, alpha = 0.72)
        } else {
          point_color_for(piece$beta, piece$opportunity, piece$model, alpha = 0.62)
        }
      }
      plot(
        NA,
        xlim = x_limits,
        ylim = y_limits,
        xlab = xlab,
        ylab = ylab,
        main = if (identical(model_value, "all")) {
          if (isTRUE(facet_sigma) && n_panels > 1L) sigma_panel_title(panel_value) else ""
        } else {
          model_panel_title(model_value, panel_value, facet_sigma = isTRUE(facet_sigma) && n_panels > 1L)
        },
        log = log_arg_for_axes(log_x, log_y)
      )
      grid()
      draw_model_points(
        piece[[x_col]],
        piece[[y_col]],
        if ("model" %in% names(piece)) piece$model else NULL,
        point_cols,
        cex = 1.35
      )
    }
  }
  par(mar = c(0, 0, 0, 0))
  if (identical(color_mode, "sigma")) {
    plot_sigma_legend(legend_model_levels)
  } else {
    plot_parameter_legend(legend_model_levels)
  }
  par(old_par)
  dev.off()
  message(sprintf("Saved %s", path))
}

plot_metric_by_difference <- function(
  summary_data,
  y_col,
  ylab,
  file_prefix,
  x_col = "node_reward_difference",
  xlab = "Node 1 actual reward - node 2 actual reward",
  log_y = FALSE
) {
  path <- safe_png_path(file_prefix, file_suffix)
  plot_data <- filter_plot_data(summary_data, x_col, y_col, log_x = FALSE, log_y = log_y, file_prefix = file_prefix)
  axis_spec <- reward_difference_axis(x_col)
  if (!is.null(axis_spec)) {
    plot_data <- plot_data[
      plot_data[[x_col]] >= axis_spec$limits[[1]] &
        plot_data[[x_col]] <= axis_spec$limits[[2]],
      ,
      drop = FALSE
    ]
  }
  if (nrow(plot_data) == 0) {
    return(invisible(NULL))
  }
  n_sigma <- max(1L, length(sigma_levels))
  model_levels <- if ("model" %in% names(summary_data)) ordered_model_levels(summary_data$model) else "VAE"
  n_models <- max(1L, length(model_levels))
  open_panel_png(path, n_cols = n_sigma, n_rows = n_models, legend_fraction = legend_panel_fraction)
  old_par <- par(no.readonly = TRUE)
  panel_matrix <- matrix(seq_len(n_sigma * n_models), nrow = n_models, ncol = n_sigma, byrow = TRUE)
  layout(cbind(panel_matrix, rep(n_sigma * n_models + 1L, n_models)),
    widths = c(rep(1, n_sigma), legend_panel_fraction)
  )
  par(mar = c(4.2, 4.2, if (n_sigma > 1L) 2 else 1, 1))
  apply_panel_text_style()
  x_limits <- if (!is.null(axis_spec)) {
    axis_spec$limits
  } else {
    expand_range(plot_data[[x_col]], pad = 0.06)
  }
  y_limits <- if (isTRUE(log_y)) {
    expand_log_range(plot_data[[y_col]], pad = 0.06)
  } else {
    expand_range(plot_data[[y_col]], pad = 0.06)
  }
  x_ticks <- if (!is.null(axis_spec)) {
    axis_spec$ticks
  } else {
    seq(floor(x_limits[[1]]), ceiling(x_limits[[2]]), by = 1)
  }
  for (model_value in model_levels) {
    model_data <- if ("model" %in% names(plot_data)) filter_model_rows(plot_data, model_value) else plot_data
    for (sigma_value in sigma_levels) {
      panel_data <- filter_sigma_rows(model_data, sigma_value)
      plot(
        NA,
        xlim = x_limits,
        ylim = y_limits,
        xlab = xlab,
        ylab = ylab,
        main = model_panel_title(model_value, sigma_value, facet_sigma = n_sigma > 1L),
        xaxt = "n",
        log = log_arg_for_axes(FALSE, log_y)
      )
      axis(1, at = x_ticks)
      grid()
      for (opportunity_value in opportunity_levels) {
        for (beta_value in beta_levels) {
          piece <- panel_data[
            parameter_value_matches(panel_data$beta, beta_value) &
              parameter_value_matches(panel_data$opportunity, opportunity_value),
            ,
            drop = FALSE
          ]
          if (nrow(piece) == 0) {
            next
          }
          piece <- piece[order(piece[[x_col]]), , drop = FALSE]
          piece_col <- line_color_for(beta_value, opportunity_value, model_value)
          lines(
            piece[[x_col]],
            piece[[y_col]],
            type = "l",
            lwd = 1.4,
            col = piece_col
          )
          draw_model_points(
            piece[[x_col]],
            piece[[y_col]],
            rep(model_value, nrow(piece)),
            rep(piece_col, nrow(piece)),
            cex = 0.85
          )
        }
      }
    }
  }
  par(mar = c(0, 0, 0, 0))
  plot_parameter_legend()
  par(old_par)
  dev.off()
  message(sprintf("Saved %s", path))
}

plot_metric_by_timestep <- function(
  summary_data,
  y_col,
  ylab,
  file_prefix,
  log_y = FALSE
) {
  path <- safe_png_path(file_prefix, file_suffix)
  plot_data <- filter_plot_data(summary_data, "timestep", y_col, log_x = FALSE, log_y = log_y, file_prefix = file_prefix)
  if (nrow(plot_data) == 0) {
    return(invisible(NULL))
  }
  n_sigma <- max(1L, length(sigma_levels))
  model_levels <- if ("model" %in% names(summary_data)) ordered_model_levels(summary_data$model) else "VAE"
  n_models <- max(1L, length(model_levels))
  open_panel_png(path, n_cols = n_sigma, n_rows = n_models, legend_fraction = legend_panel_fraction)
  old_par <- par(no.readonly = TRUE)
  panel_matrix <- matrix(seq_len(n_sigma * n_models), nrow = n_models, ncol = n_sigma, byrow = TRUE)
  layout(cbind(panel_matrix, rep(n_sigma * n_models + 1L, n_models)),
    widths = c(rep(1, n_sigma), legend_panel_fraction)
  )
  par(mar = c(4.2, 4.2, if (n_sigma > 1L) 2 else 1, 1))
  apply_panel_text_style()
  x_values <- suppressWarnings(as.numeric(plot_data$timestep))
  x_limits <- expand_range(x_values, pad = 0.08)
  x_ticks <- seq(floor(min(x_values, na.rm = TRUE)), ceiling(max(x_values, na.rm = TRUE)), by = 1)
  y_limits <- if (isTRUE(log_y)) {
    expand_log_range(plot_data[[y_col]], pad = 0.06)
  } else {
    expand_range(plot_data[[y_col]], pad = 0.06)
  }
  for (model_value in model_levels) {
    model_data <- if ("model" %in% names(plot_data)) filter_model_rows(plot_data, model_value) else plot_data
    for (sigma_value in sigma_levels) {
      panel_data <- filter_sigma_rows(model_data, sigma_value)
      plot(
        NA,
        xlim = x_limits,
        ylim = y_limits,
        xlab = "Pre-stop observation timestep",
        ylab = ylab,
        main = model_panel_title(model_value, sigma_value, facet_sigma = n_sigma > 1L),
        xaxt = "n",
        log = log_arg_for_axes(FALSE, log_y)
      )
      axis(1, at = x_ticks)
      grid()
      for (opportunity_value in opportunity_levels) {
        for (beta_value in beta_levels) {
          piece <- panel_data[
            parameter_value_matches(panel_data$beta, beta_value) &
              parameter_value_matches(panel_data$opportunity, opportunity_value),
            ,
            drop = FALSE
          ]
          if (nrow(piece) == 0) {
            next
          }
          piece <- piece[order(piece$timestep), , drop = FALSE]
          piece_col <- line_color_for(beta_value, opportunity_value, model_value)
          lines(
            piece$timestep,
            piece[[y_col]],
            type = "l",
            lwd = 1.4,
            col = piece_col
          )
          draw_model_points(
            piece$timestep,
            piece[[y_col]],
            rep(model_value, nrow(piece)),
            rep(piece_col, nrow(piece)),
            cex = 0.85
          )
        }
      }
    }
  }
  par(mar = c(0, 0, 0, 0))
  plot_parameter_legend()
  par(old_par)
  dev.off()
  message(sprintf("Saved %s", path))
}

coerce_chosen_path_to_one_based <- function(chosen_path, model = NULL) {
  chosen <- suppressWarnings(as.integer(chosen_path))
  if (is.null(model)) {
    finite_chosen <- chosen[!is.na(chosen)]
    if (length(finite_chosen) > 0 && min(finite_chosen) == 0L && max(finite_chosen) <= (task_path_count - 1L)) {
      chosen <- chosen + 1L
    }
    return(chosen)
  }
  model <- as.character(model)
  zero_based_rows <- !is.na(chosen) &
    model != "Optimal" &
    chosen >= 0L &
    chosen <= (task_path_count - 1L)
  if (any(zero_based_rows)) {
    chosen[zero_based_rows] <- chosen[zero_based_rows] + 1L
  }
  chosen
}

build_node_difference_trial_metrics <- function(trial_data, node_mat) {
  if (task_path_count < 2L) {
    warning("Task has fewer than two paths; skipping path-difference revisit plots.")
    return(data.frame())
  }
  if (!"chosen_path" %in% names(trial_data)) {
    warning("Missing chosen_path column; skipping path-difference revisit plots.")
    return(data.frame())
  }
  actual_lookup <- build_node_actual_reward_lookup(all_data)
  if (!isTRUE(actual_lookup$available)) {
    warning("Missing node/actual_reward columns; skipping path-difference revisit plots.")
    return(data.frame())
  }
  actual_reward_mat <- sapply(seq_len(task_node_count), function(node_id) {
    actual_reward_for_nodes(trial_data, rep(node_id, nrow(trial_data)), actual_lookup)
  })
  if (is.null(dim(actual_reward_mat))) {
    actual_reward_mat <- matrix(actual_reward_mat, nrow = nrow(trial_data), ncol = task_node_count)
  }
  valid_actual_reward_mat <- matrix(
    is_valid_task_reward(as.vector(actual_reward_mat)),
    nrow = nrow(actual_reward_mat),
    ncol = ncol(actual_reward_mat)
  )
  valid_actual_rewards <- rowSums(valid_actual_reward_mat) == ncol(valid_actual_reward_mat)
  if (any(!valid_actual_rewards)) {
    warning(sprintf(
      "Dropping %d rows with impossible node rewards from node-difference revisit plots.",
      sum(!valid_actual_rewards)
    ))
    actual_reward_mat[!valid_actual_rewards, ] <- NA_real_
  }
  path_nodes <- task_path_nodes()
  path_reward_mat <- sapply(path_nodes, function(nodes) {
    rowSums(actual_reward_mat[, nodes, drop = FALSE])
  })
  if (is.null(dim(path_reward_mat))) {
    path_reward_mat <- matrix(path_reward_mat, nrow = nrow(trial_data), ncol = task_path_count)
  }
  stop_timestep <- suppressWarnings(as.numeric(trial_data$stop_decision_timestep))
  timestep_before_stop <- suppressWarnings(as.numeric(trial_data$timestep_before_stop))
  path_mat <- if (ncol(node_mat) > 0L) {
    matrix(
      path_id_for_node(as.vector(node_mat)),
      nrow = nrow(node_mat),
      ncol = ncol(node_mat)
    )
  } else {
    matrix(numeric(), nrow = nrow(trial_data), ncol = 0L)
  }
  if (ncol(node_mat) > 0L) {
    observation_col_indices <- matrix(
      seq_len(ncol(node_mat)),
      nrow = nrow(node_mat),
      ncol = ncol(node_mat),
      byrow = TRUE
    )
    pre_stop_observation_mat <- observation_col_indices <= timestep_before_stop
    pre_stop_observation_mat[is.na(pre_stop_observation_mat)] <- FALSE
  } else {
    pre_stop_observation_mat <- matrix(FALSE, nrow = nrow(trial_data), ncol = 0L)
  }
  observations <- rowSums(is.finite(path_mat) & pre_stop_observation_mat, na.rm = TRUE)
  path_1_visits <- rowSums((path_mat == 1) & pre_stop_observation_mat, na.rm = TRUE)

  row_mean_excluding_path <- function(reward_matrix, exclude_path) {
    out <- rep(NA_real_, nrow(reward_matrix))
    exclude_path <- suppressWarnings(as.integer(exclude_path))
    valid <- !is.na(exclude_path) & exclude_path >= 1L & exclude_path <= ncol(reward_matrix)
    for (path_id in seq_len(ncol(reward_matrix))) {
      rows <- valid & exclude_path == path_id
      if (!any(rows)) {
        next
      }
      other_cols <- setdiff(seq_len(ncol(reward_matrix)), path_id)
      out[rows] <- rowMeans(reward_matrix[rows, other_cols, drop = FALSE])
    }
    out
  }
  path_1_reward <- path_reward_mat[, 1L]
  path_1_mean_other_reward <- rowMeans(path_reward_mat[, -1L, drop = FALSE])

  finite_path_mat <- is.finite(path_mat) & pre_stop_observation_mat
  if (ncol(path_mat) > 0L && any(finite_path_mat)) {
    path_col_indices <- matrix(
      seq_len(ncol(path_mat)),
      nrow = nrow(path_mat),
      ncol = ncol(path_mat),
      byrow = TRUE
    )
    last_path_col <- max.col(ifelse(finite_path_mat, path_col_indices, 0), ties.method = "first")
    has_path <- rowSums(finite_path_mat) > 0
    last_path <- rep(NA_real_, nrow(path_mat))
    last_path[has_path] <- path_mat[cbind(which(has_path), last_path_col[has_path])]
  } else {
    last_path <- rep(NA_real_, nrow(trial_data))
  }
  last_path_int <- suppressWarnings(as.integer(last_path))
  last_visited_path_reward <- rep(NA_real_, nrow(trial_data))
  valid_last_path <- !is.na(last_path_int) & last_path_int >= 1L & last_path_int <= ncol(path_reward_mat)
  if (any(valid_last_path)) {
    last_rows <- which(valid_last_path)
    last_visited_path_reward[last_rows] <- path_reward_mat[cbind(last_rows, last_path_int[last_rows])]
  }
  other_mean_after_last_visit <- row_mean_excluding_path(path_reward_mat, last_path_int)
  path_1_minus_mean_other_path_raw <- path_1_reward - path_1_mean_other_reward
  last_visited_minus_mean_other_path_raw <- last_visited_path_reward - other_mean_after_last_visit
  chosen_path <- coerce_chosen_path_to_one_based(trial_data$chosen_path, trial_data$model)
  trial_metrics <- data.frame(
    model = trial_data$model,
    sigma = trial_data$sigma,
    beta = trial_data$beta,
    opportunity = trial_data$opportunity,
    path_reward_difference = path_1_minus_mean_other_path_raw,
    absolute_path_reward_difference = abs(path_1_minus_mean_other_path_raw),
    last_visited_path_reward_difference = last_visited_minus_mean_other_path_raw,
    path_1_minus_mean_other_path = bin_mean_difference(path_1_minus_mean_other_path_raw),
    absolute_path_1_minus_mean_other_path = bin_mean_difference(abs(path_1_minus_mean_other_path_raw)),
    last_visited_minus_mean_other_path = bin_mean_difference(last_visited_minus_mean_other_path_raw),
    proportion_timesteps_path1 = ifelse(observations > 0, path_1_visits / observations, NA_real_),
    last_visited_path_chosen = ifelse(is.finite(last_path) & !is.na(chosen_path), as.numeric(last_path == chosen_path), NA_real_),
    choose_path1 = ifelse(!is.na(chosen_path), as.numeric(chosen_path == 1L), NA_real_),
    node_reward_difference = path_1_minus_mean_other_path_raw,
    absolute_node_reward_difference = abs(path_1_minus_mean_other_path_raw),
    last_visited_reward_difference = last_visited_minus_mean_other_path_raw,
    node_1_minus_mean_other_node = bin_mean_difference(path_1_minus_mean_other_path_raw),
    absolute_node_1_minus_mean_other_node = bin_mean_difference(abs(path_1_minus_mean_other_path_raw)),
    last_visited_minus_mean_other_node = bin_mean_difference(last_visited_minus_mean_other_path_raw),
    proportion_timesteps_node1 = ifelse(observations > 0, path_1_visits / observations, NA_real_),
    last_visited_node_chosen = ifelse(is.finite(last_path) & !is.na(chosen_path), as.numeric(last_path == chosen_path), NA_real_),
    choose_node1 = ifelse(!is.na(chosen_path), as.numeric(chosen_path == 1L), NA_real_),
    observations_before_stop = observations,
    stop_decision_timestep = stop_timestep,
    timestep_before_stop = timestep_before_stop,
    kl_paid_total = suppressWarnings(as.numeric(trial_data$kl_paid_total)),
    kl_paid_per_stop_timestep = ifelse(
      is.finite(timestep_before_stop) & timestep_before_stop > 0,
      suppressWarnings(as.numeric(trial_data$kl_paid_total)) / timestep_before_stop,
      NA_real_
    ),
    stringsAsFactors = FALSE
  )
  trial_metrics[
    is.finite(trial_metrics$path_1_minus_mean_other_path) |
      is.finite(trial_metrics$absolute_path_1_minus_mean_other_path) |
      is.finite(trial_metrics$last_visited_minus_mean_other_path) |
      is.finite(trial_metrics$node_reward_difference) |
      is.finite(trial_metrics$absolute_node_reward_difference) |
      is.finite(trial_metrics$last_visited_reward_difference),
    ,
    drop = FALSE
  ]
}

summarize_difference_metrics <- function(trial_metrics, difference_col) {
  if (nrow(trial_metrics) == 0 || !difference_col %in% names(trial_metrics)) {
    return(data.frame())
  }
  trial_metrics <- trial_metrics[is.finite(trial_metrics[[difference_col]]), , drop = FALSE]
  if (nrow(trial_metrics) == 0) {
    return(data.frame())
  }
  group_cols <- c("model", "sigma", "beta", "opportunity", difference_col)
  out <- aggregate_means_by(
    trial_metrics,
    group_cols = group_cols,
    value_cols = c(
      "proportion_timesteps_path1",
      "last_visited_path_chosen",
      "choose_path1",
      "observations_before_stop",
      "stop_decision_timestep",
      "timestep_before_stop",
      "kl_paid_total",
      "kl_paid_per_stop_timestep"
    )
  )
  count_data <- count_rows_by(trial_metrics, group_cols)
  merge(out, count_data, by = group_cols, all.x = TRUE)
}

plot_summary_scatter(
  average_summary,
  y_col = "kl_paid_total",
  x_col = "timestep_before_stop",
  xlab = "Average timestep before stopping",
  ylab = "Average KL paid across timesteps (log)",
  file_prefix = "revisit_log_kl_paid_vs_average_timestep_before_stop_by_sigma",
  log_y = TRUE,
  facet_sigma = TRUE,
  color_mode = "parameter",
  model_layout = "overlay"
)
plot_summary_scatter(
  average_summary,
  x_col = "normalized_chosen_path_reward",
  y_col = "timestep_before_stop",
  xlab = "Average normalized chosen path reward",
  ylab = "Average timestep before stopping",
  file_prefix = "revisit_average_timestep_before_stop_vs_average_normalized_chosen_path_reward_by_sigma",
  facet_sigma = TRUE,
  color_mode = "parameter",
  model_layout = "overlay"
)
plot_summary_scatter(
  average_summary,
  x_col = "normalized_chosen_path_reward",
  y_col = "kl_paid_total",
  xlab = "Average normalized chosen path reward",
  ylab = "Average KL paid across timesteps (log)",
  file_prefix = "revisit_log_kl_paid_vs_average_normalized_chosen_path_reward_by_sigma",
  log_y = TRUE,
  facet_sigma = TRUE,
  color_mode = "parameter",
  model_layout = "overlay"
)

plot_summary_scatter(
  average_summary,
  x_col = "timestep_before_stop",
  y_col = "kl_paid_total",
  xlab = "Average timestep before stopping",
  ylab = "Average KL paid across timesteps (log)",
  file_prefix = "revisit_log_kl_paid_vs_average_timestep_before_stop_sigma_color",
  log_y = TRUE,
  facet_sigma = FALSE,
  color_mode = "sigma",
  model_layout = "overlay"
)
plot_summary_scatter(
  average_summary,
  x_col = "normalized_chosen_path_reward",
  y_col = "timestep_before_stop",
  xlab = "Average normalized chosen path reward",
  ylab = "Average timestep before stopping",
  file_prefix = "revisit_average_timestep_before_stop_vs_average_normalized_chosen_path_reward_sigma_color",
  facet_sigma = FALSE,
  color_mode = "sigma",
  model_layout = "overlay"
)
plot_summary_scatter(
  average_summary,
  x_col = "normalized_chosen_path_reward",
  y_col = "kl_paid_total",
  xlab = "Average normalized chosen path reward",
  ylab = "Average KL paid across timesteps (log)",
  file_prefix = "revisit_log_kl_paid_vs_average_normalized_chosen_path_reward_sigma_color",
  log_y = TRUE,
  facet_sigma = FALSE,
  color_mode = "sigma",
  model_layout = "overlay"
)

entropy_summary_col <- if (
  "terminal_choice_entropy_combined_reached" %in% names(average_summary) &&
    any(is.finite(suppressWarnings(as.numeric(average_summary$terminal_choice_entropy_combined_reached))))
) {
  "terminal_choice_entropy_combined_reached"
} else {
  "terminal_choice_entropy"
}
entropy_summary_label <- if (identical(entropy_summary_col, "terminal_choice_entropy_combined_reached")) {
  "Average stop-choice entropy across reached timesteps"
} else {
  "Average terminal choice entropy at stop"
}
has_entropy_data <- entropy_summary_col %in% names(average_summary) &&
  any(is.finite(suppressWarnings(as.numeric(average_summary[[entropy_summary_col]]))))
if (has_entropy_data) {
  plot_summary_scatter(
    average_summary,
    x_col = "kl_paid_total",
    y_col = entropy_summary_col,
    xlab = "Average KL paid across timesteps (log)",
    ylab = entropy_summary_label,
    file_prefix = "revisit_average_terminal_choice_entropy_vs_log_kl_paid_by_sigma",
    log_x = TRUE,
    facet_sigma = TRUE,
    color_mode = "parameter",
    model_layout = "overlay"
  )
  plot_summary_scatter(
    average_summary,
    x_col = "timestep_before_stop",
    y_col = entropy_summary_col,
    xlab = "Average timestep before stopping",
    ylab = entropy_summary_label,
    file_prefix = "revisit_average_terminal_choice_entropy_vs_average_timestep_before_stop_by_sigma",
    facet_sigma = TRUE,
    color_mode = "parameter",
    model_layout = "overlay"
  )
  plot_summary_scatter(
    average_summary,
    x_col = "kl_paid_total",
    y_col = entropy_summary_col,
    xlab = "Average KL paid across timesteps (log)",
    ylab = entropy_summary_label,
    file_prefix = "revisit_average_terminal_choice_entropy_vs_log_kl_paid_sigma_color",
    log_x = TRUE,
    facet_sigma = FALSE,
    color_mode = "sigma",
    model_layout = "overlay"
  )
  plot_summary_scatter(
    average_summary,
    x_col = "timestep_before_stop",
    y_col = entropy_summary_col,
    xlab = "Average timestep before stopping",
    ylab = entropy_summary_label,
    file_prefix = "revisit_average_terminal_choice_entropy_vs_average_timestep_before_stop_sigma_color",
    facet_sigma = FALSE,
    color_mode = "sigma",
    model_layout = "overlay"
  )
} else {
  warning(
    "No terminal choice probability columns found in the simulation CSVs; skipping terminal-choice entropy plots."
  )
}

plot_metric_by_timestep(
  pre_stop_timestep_summary,
  y_col = "kl_paid_at_timestep",
  ylab = "Average KL paid at timestep",
  file_prefix = "revisit_pre_stop_timestep_vs_kl_paid_at_timestep_by_sigma"
)
has_pre_stop_entropy_data <- nrow(pre_stop_timestep_summary) > 0 &&
  "terminal_binary_choice_entropy_at_timestep" %in% names(pre_stop_timestep_summary) &&
  any(is.finite(suppressWarnings(as.numeric(pre_stop_timestep_summary$terminal_binary_choice_entropy_at_timestep))))
if (has_pre_stop_entropy_data) {
  plot_metric_by_timestep(
    pre_stop_timestep_summary,
    y_col = "terminal_binary_choice_entropy_at_timestep",
    ylab = "Average stop-choice entropy, path 1 vs path 2",
    file_prefix = "revisit_pre_stop_timestep_vs_terminal_choice_entropy_at_timestep_by_sigma"
  )
} else {
  warning(
    "No terminal choice probability columns found at pre-stop timesteps; skipping pre-stop terminal-choice entropy plot."
  )
}

node_difference_trial_metrics <- build_node_difference_trial_metrics(trial_data, node_mat)
if (nrow(node_difference_trial_metrics) > 0) {
  node_difference_summary <- summarize_difference_metrics(
    node_difference_trial_metrics,
    "path_1_minus_mean_other_path"
  )
  last_visited_difference_summary <- summarize_difference_metrics(
    node_difference_trial_metrics,
    "last_visited_minus_mean_other_path"
  )
  absolute_node_difference_summary <- summarize_difference_metrics(
    node_difference_trial_metrics,
    "absolute_path_1_minus_mean_other_path"
  )
  plot_metric_by_difference(
    node_difference_summary,
    y_col = "proportion_timesteps_path1",
    ylab = "Proportion of pre-stop timesteps visiting path 1",
    file_prefix = "revisit_path1_minus_mean_other_path_reward_vs_proportion_path1_visits_by_sigma",
    x_col = "path_1_minus_mean_other_path",
    xlab = "Path 1 actual reward - mean actual other path(s)"
  )
  plot_metric_by_difference(
    last_visited_difference_summary,
    y_col = "last_visited_path_chosen",
    ylab = "P(last visited path is chosen)",
    file_prefix = "revisit_last_visited_minus_mean_other_path_reward_vs_last_visited_path_chosen_by_sigma",
    x_col = "last_visited_minus_mean_other_path",
    xlab = "Last visited path actual reward - mean actual other path(s)"
  )
  plot_metric_by_difference(
    node_difference_summary,
    y_col = "choose_path1",
    ylab = "P(choose path 1)",
    file_prefix = "revisit_path1_minus_mean_other_path_reward_vs_probability_choose_path1_by_sigma",
    x_col = "path_1_minus_mean_other_path",
    xlab = "Path 1 actual reward - mean actual other path(s)"
  )
  plot_metric_by_difference(
    absolute_node_difference_summary,
    y_col = "timestep_before_stop",
    ylab = "Average timestep before stopping",
    file_prefix = "revisit_abs_path1_minus_mean_other_path_reward_vs_average_timestep_before_stop_by_sigma",
    x_col = "absolute_path_1_minus_mean_other_path",
    xlab = "|path 1 actual reward - mean actual other path(s)|"
  )
  plot_metric_by_difference(
    absolute_node_difference_summary,
    y_col = "kl_paid_total",
    ylab = "Average KL paid across timesteps",
    file_prefix = "revisit_abs_path1_minus_mean_other_path_reward_vs_average_kl_paid_by_sigma",
    x_col = "absolute_path_1_minus_mean_other_path",
    xlab = "|path 1 actual reward - mean actual other path(s)|"
  )
  plot_metric_by_difference(
    absolute_node_difference_summary,
    y_col = "kl_paid_per_stop_timestep",
    ylab = "Average KL paid per stop timestep",
    file_prefix = "revisit_abs_path1_minus_mean_other_path_reward_vs_average_kl_paid_per_timestep_before_stop_by_sigma",
    x_col = "absolute_path_1_minus_mean_other_path",
    xlab = "|path 1 actual reward - mean actual other path(s)|"
  )
}
