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

dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)
message(sprintf("Using %s revisit simulation CSVs from %s", simulation_source, input_dir))

tree_file_label <- paste0(tree_size, "n", if (nzchar(tree_config)) paste0("_", tree_config) else "")
architecture_file_label <- sprintf("rnn_%s_latent_%s", rnn_units_arg, latent_dim_arg)
revisit_label <- paste0("revisit_maxobs_", max_observations_arg)
source_suffix <- if (identical(simulation_source, "jax")) "_source_jax" else ""

beta_values <- trimws(strsplit(beta_arg, ",")[[1]])
opportunity_values <- trimws(strsplit(opportunity_arg, ",")[[1]])
sigma_values <- trimws(strsplit(sigma_arg, ",")[[1]])
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
sigma_label <- compact_arg_label(sigma_values)
sigma_numeric_values <- suppressWarnings(as.numeric(sigma_values))
sigma_arg_is_default <- length(sigma_values) == 1L &&
  !is.na(sigma_numeric_values[[1]]) &&
  abs(sigma_numeric_values[[1]]) < 1e-12
sigma_file_suffix <- if (sigma_arg_is_default) "" else paste0("_sigma_", sigma_label)
file_suffix <- sprintf(
  "%s_lambda_%s_alpha_%s_beta_%s_opportunity_%s_expansion_%s_variant_%s_%s_%s_%s%s%s",
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

safe_png_path <- function(file_prefix, suffix, max_basename_chars = 240) {
  basename <- sprintf("%s_%s.png", file_prefix, suffix)
  if (nchar(basename, type = "bytes") <= max_basename_chars) {
    return(file.path(results_dir, basename))
  }
  hash <- short_string_hash(basename)
  fixed_chars <- nchar(file_prefix, type = "bytes") +
    nchar(hash, type = "bytes") +
    nchar("_h.png", type = "bytes") +
    1L
  keep_chars <- max(16L, max_basename_chars - fixed_chars)
  compact_suffix <- substr(suffix, 1L, keep_chars)
  file.path(results_dir, sprintf("%s_%s_h%s.png", file_prefix, compact_suffix, hash))
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

all_data <- bind_rows_fill(loaded_data)
if (is.null(all_data) || nrow(all_data) == 0) {
  stop("No revisit simulation CSVs were found. Check parameters, seeds, input_dir, and max observations.")
}

if ("opportunity_cost" %in% names(all_data)) {
  requested_opportunity <- suppressWarnings(as.numeric(opportunity_values))
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
    return(x)
  }
  tolower(as.character(x)) %in% c("true", "t", "1", "yes", "y", "stop")
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
  key_cols <- intersect(c("sigma", "beta", "opportunity", "seed", "graph"), names(dat))
  if (!all(c("node", "actual_reward") %in% names(dat))) {
    return(list(key_cols = key_cols, rewards = numeric(), available = FALSE))
  }
  node_data <- unique(dat[, unique(c(key_cols, "node", "actual_reward")), drop = FALSE])
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
  trial_id_cols <- intersect(c("sigma", "beta", "opportunity", "seed", "graph"), names(dat))
  trial_cols <- unique(c(trial_id_cols, required_cols))
  unique(dat[, trial_cols, drop = FALSE])
}

trial_id_cols <- intersect(c("sigma", "beta", "opportunity", "seed", "graph"), names(all_data))
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
  has_explicit_stop <- apply(stop_mat, 1L, any)
  first_stop_timestep <- apply(stop_mat, 1L, function(row) {
    idx <- which(row)
    if (length(idx) == 0) NA_real_ else stop_timesteps[[idx[[1]]]]
  })
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
trial_data$unique_nodes_visited <- apply(node_mat, 1, function(row) {
  length(unique(row[is.finite(row)]))
})
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
trial_data <- trial_data[is.finite(trial_data$normalized_chosen_path_reward), , drop = FALSE]

mean_or_na <- function(x) {
  x <- suppressWarnings(as.numeric(x))
  x <- x[is.finite(x)]
  if (length(x) == 0) NA_real_ else mean(x)
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
      sigma = trial_data$sigma[before_stop],
      beta = trial_data$beta[before_stop],
      opportunity = trial_data$opportunity[before_stop],
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
  pieces <- lapply(value_cols, function(value_col) {
    out <- aggregate(dat[[value_col]], by = dat[, group_cols, drop = FALSE], FUN = mean_or_na)
    names(out)[names(out) == "x"] <- value_col
    out
  })
  Reduce(function(left, right) merge(left, right, by = group_cols, all = TRUE), pieces)
}

pre_stop_timestep_data <- build_pre_stop_timestep_data()
if (nrow(pre_stop_timestep_data) > 0) {
  pre_stop_timestep_summary <- aggregate_means_by(
    pre_stop_timestep_data,
    group_cols = c("sigma", "beta", "opportunity", "timestep"),
    value_cols = c(
      "kl_paid_at_timestep",
      "terminal_binary_choice_entropy_at_timestep"
    )
  )
  pre_stop_timestep_count <- aggregate(
    list(n = rep(1, nrow(pre_stop_timestep_data))),
    by = pre_stop_timestep_data[, c("sigma", "beta", "opportunity", "timestep"), drop = FALSE],
    FUN = sum
  )
  pre_stop_timestep_summary <- merge(
    pre_stop_timestep_summary,
    pre_stop_timestep_count,
    by = c("sigma", "beta", "opportunity", "timestep"),
    all.x = TRUE
  )
} else {
  pre_stop_timestep_summary <- data.frame()
}
if (nrow(pre_stop_timestep_data) > 0) {
  pre_stop_entropy_combo_summary <- aggregate_means_by(
    pre_stop_timestep_data,
    group_cols = c("sigma", "beta", "opportunity"),
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
  group_cols = c("sigma", "beta", "opportunity"),
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
count_data <- aggregate(
  normalized_chosen_path_reward ~ sigma + beta + opportunity,
  data = trial_data,
  FUN = length
)
names(count_data)[names(count_data) == "normalized_chosen_path_reward"] <- "n"
average_summary <- merge(average_summary, count_data, by = c("sigma", "beta", "opportunity"), all.x = TRUE)
if (nrow(pre_stop_entropy_combo_summary) > 0) {
  average_summary <- merge(
    average_summary,
    pre_stop_entropy_combo_summary,
    by = c("sigma", "beta", "opportunity"),
    all.x = TRUE
  )
}

beta_levels <- beta_values[beta_values %in% unique(all_data$beta)]
if (length(beta_levels) == 0) {
  beta_levels <- unique(all_data$beta)
}
opportunity_levels <- opportunity_values[opportunity_values %in% unique(all_data$opportunity)]
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

color_by <- if (length(opportunity_levels) > 1 && length(beta_levels) == 1) "opportunity" else "beta"
color_levels <- if (identical(color_by, "opportunity")) opportunity_levels else beta_levels
color_cols <- if (identical(color_by, "opportunity")) {
  numeric_gradient_colors(color_levels, "Blues", darker_high = TRUE)
} else {
  numeric_gradient_colors(color_levels, "Greens", darker_high = FALSE)
}

series_color <- function(beta_value, opportunity_value) {
  if (identical(color_by, "opportunity")) {
    unname(color_cols[as.character(opportunity_value)])
  } else {
    unname(color_cols[as.character(beta_value)])
  }
}

point_color_for <- function(beta, opportunity, alpha = 0.55) {
  cols <- mapply(series_color, beta, opportunity)
  grDevices::adjustcolor(cols, alpha.f = alpha)
}

line_color_for <- function(beta, opportunity) {
  mapply(series_color, beta, opportunity)
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

plot_parameter_legend <- function() {
  plot.new()
  legend_title <- if (identical(color_by, "opportunity")) "opportunity" else "beta"
  legend(
    "center",
    legend = paste(legend_title, format_plot_values(color_levels)),
    col = unname(color_cols[color_levels]),
    pch = 19,
    lty = 1,
    bty = "n",
    cex = 0.9
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

plot_sigma_legend <- function() {
  plot.new()
  legend(
    "center",
    legend = paste("sigma", format_plot_values(sigma_levels)),
    col = unname(sigma_cols[sigma_levels]),
    pch = 19,
    lty = 1,
    bty = "n",
    cex = 0.9
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
  color_mode = c("parameter", "sigma")
) {
  color_mode <- match.arg(color_mode)
  path <- safe_png_path(file_prefix, file_suffix)
  plot_data <- filter_plot_data(summary_data, x_col, y_col, log_x, log_y, file_prefix)
  if (nrow(plot_data) == 0) {
    return(invisible(NULL))
  }
  panel_levels <- if (isTRUE(facet_sigma)) sigma_levels else "all"
  n_panels <- max(1L, length(panel_levels))
  open_panel_png(path, n_cols = n_panels, n_rows = 1L, legend_fraction = legend_panel_fraction)
  old_par <- par(no.readonly = TRUE)
  layout(matrix(c(seq_len(n_panels), n_panels + 1L), nrow = 1), widths = c(rep(1, n_panels), legend_panel_fraction))
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
  for (panel_value in panel_levels) {
    piece <- if (isTRUE(facet_sigma)) filter_sigma_rows(plot_data, panel_value) else plot_data
    point_cols <- character()
    if (nrow(piece) > 0) {
      point_cols <- if (identical(color_mode, "sigma")) {
        sigma_color_for(piece$sigma, alpha = 0.72)
      } else {
        point_color_for(piece$beta, piece$opportunity, alpha = 0.62)
      }
    }
    plot(
      NA,
      xlim = x_limits,
      ylim = y_limits,
      xlab = xlab,
      ylab = ylab,
      main = if (isTRUE(facet_sigma) && n_panels > 1L) sigma_panel_title(panel_value) else "",
      log = log_arg_for_axes(log_x, log_y)
    )
    grid()
    points(
      piece[[x_col]],
      piece[[y_col]],
      pch = 19,
      cex = 1.35,
      col = point_cols
    )
  }
  par(mar = c(0, 0, 0, 0))
  if (identical(color_mode, "sigma")) {
    plot_sigma_legend()
  } else {
    plot_parameter_legend()
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
  if (nrow(plot_data) == 0) {
    return(invisible(NULL))
  }
  n_sigma <- max(1L, length(sigma_levels))
  open_panel_png(path, n_cols = n_sigma, n_rows = 1L, legend_fraction = legend_panel_fraction)
  old_par <- par(no.readonly = TRUE)
  layout(matrix(c(seq_len(n_sigma), n_sigma + 1L), nrow = 1), widths = c(rep(1, n_sigma), legend_panel_fraction))
  par(mar = c(4.2, 4.2, if (n_sigma > 1L) 2 else 1, 1))
  apply_panel_text_style()
  x_limits <- expand_range(plot_data[[x_col]], pad = 0.06)
  y_limits <- if (isTRUE(log_y)) {
    expand_log_range(plot_data[[y_col]], pad = 0.06)
  } else {
    expand_range(plot_data[[y_col]], pad = 0.06)
  }
  x_ticks <- seq(floor(x_limits[[1]]), ceiling(x_limits[[2]]), by = 1)
  for (sigma_value in sigma_levels) {
    panel_data <- filter_sigma_rows(plot_data, sigma_value)
    plot(
      NA,
      xlim = x_limits,
      ylim = y_limits,
      xlab = xlab,
      ylab = ylab,
      main = if (n_sigma > 1L) sigma_panel_title(sigma_value) else "",
      xaxt = "n",
      log = log_arg_for_axes(FALSE, log_y)
    )
    axis(1, at = x_ticks)
    grid()
    for (opportunity_value in opportunity_levels) {
      for (beta_value in beta_levels) {
        piece <- panel_data[
          panel_data$beta == beta_value &
            panel_data$opportunity == opportunity_value,
          ,
          drop = FALSE
        ]
        if (nrow(piece) == 0) {
          next
        }
        piece <- piece[order(piece[[x_col]]), , drop = FALSE]
        lines(
          piece[[x_col]],
          piece[[y_col]],
          type = "b",
          pch = 19,
          lwd = 1.4,
          cex = 0.85,
          col = line_color_for(beta_value, opportunity_value)
        )
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
  open_panel_png(path, n_cols = n_sigma, n_rows = 1L, legend_fraction = legend_panel_fraction)
  old_par <- par(no.readonly = TRUE)
  layout(matrix(c(seq_len(n_sigma), n_sigma + 1L), nrow = 1), widths = c(rep(1, n_sigma), legend_panel_fraction))
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
  for (sigma_value in sigma_levels) {
    panel_data <- filter_sigma_rows(plot_data, sigma_value)
    plot(
      NA,
      xlim = x_limits,
      ylim = y_limits,
      xlab = "Pre-stop observation timestep",
      ylab = ylab,
      main = if (n_sigma > 1L) sigma_panel_title(sigma_value) else "",
      xaxt = "n",
      log = log_arg_for_axes(FALSE, log_y)
    )
    axis(1, at = x_ticks)
    grid()
    for (opportunity_value in opportunity_levels) {
      for (beta_value in beta_levels) {
        piece <- panel_data[
          panel_data$beta == beta_value &
            panel_data$opportunity == opportunity_value,
          ,
          drop = FALSE
        ]
        if (nrow(piece) == 0) {
          next
        }
        piece <- piece[order(piece$timestep), , drop = FALSE]
        lines(
          piece$timestep,
          piece[[y_col]],
          type = "b",
          pch = 19,
          lwd = 1.4,
          cex = 0.85,
          col = line_color_for(beta_value, opportunity_value)
        )
      }
    }
  }
  par(mar = c(0, 0, 0, 0))
  plot_parameter_legend()
  par(old_par)
  dev.off()
  message(sprintf("Saved %s", path))
}

coerce_chosen_path_to_one_based <- function(chosen_path) {
  chosen <- suppressWarnings(as.integer(chosen_path))
  finite_chosen <- chosen[!is.na(chosen)]
  if (length(finite_chosen) > 0 && min(finite_chosen) == 0L && max(finite_chosen) <= (task_path_count - 1L)) {
    chosen <- chosen + 1L
  }
  chosen
}

last_finite_value <- function(row) {
  values <- row[is.finite(row)]
  if (length(values) == 0) NA_real_ else values[[length(values)]]
}

build_node_difference_trial_metrics <- function(trial_data, node_mat) {
  if (task_node_count < 2L) {
    warning("Task has fewer than two nodes; skipping node-difference revisit plots.")
    return(data.frame())
  }
  if (task_path_count != 2L || task_nodes_per_path != 1L) {
    warning("Node-difference choice plots are only unambiguous for two one-node paths; skipping.")
    return(data.frame())
  }
  if (!"chosen_path" %in% names(trial_data)) {
    warning("Missing chosen_path column; skipping node-difference revisit plots.")
    return(data.frame())
  }
  actual_lookup <- build_node_actual_reward_lookup(all_data)
  if (!isTRUE(actual_lookup$available)) {
    warning("Missing node/actual_reward columns; skipping node-difference revisit plots.")
    return(data.frame())
  }
  node_1_reward <- actual_reward_for_nodes(trial_data, rep(1L, nrow(trial_data)), actual_lookup)
  node_2_reward <- actual_reward_for_nodes(trial_data, rep(2L, nrow(trial_data)), actual_lookup)
  node_1_visits <- rowSums(node_mat == 1, na.rm = TRUE)
  observations <- suppressWarnings(as.numeric(trial_data$observations_before_stop))
  stop_timestep <- suppressWarnings(as.numeric(trial_data$stop_decision_timestep))
  timestep_before_stop <- suppressWarnings(as.numeric(trial_data$timestep_before_stop))
  last_node <- apply(node_mat, 1L, last_finite_value)
  last_node_int <- suppressWarnings(as.integer(last_node))
  last_visited_reward <- ifelse(
    last_node_int == 1L,
    node_1_reward,
    ifelse(last_node_int == 2L, node_2_reward, NA_real_)
  )
  other_reward_after_last_visit <- ifelse(
    last_node_int == 1L,
    node_2_reward,
    ifelse(last_node_int == 2L, node_1_reward, NA_real_)
  )
  chosen_path <- coerce_chosen_path_to_one_based(trial_data$chosen_path)
  trial_metrics <- data.frame(
    sigma = trial_data$sigma,
    beta = trial_data$beta,
    opportunity = trial_data$opportunity,
    node_reward_difference = node_1_reward - node_2_reward,
    absolute_node_reward_difference = abs(node_1_reward - node_2_reward),
    last_visited_reward_difference = last_visited_reward - other_reward_after_last_visit,
    proportion_timesteps_node1 = ifelse(observations > 0, node_1_visits / observations, NA_real_),
    last_visited_node_chosen = ifelse(is.finite(last_node) & !is.na(chosen_path), as.numeric(last_node == chosen_path), NA_real_),
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
  group_cols <- c("sigma", "beta", "opportunity", difference_col)
  out <- aggregate_means_by(
    trial_metrics,
    group_cols = group_cols,
    value_cols = c(
      "proportion_timesteps_node1",
      "last_visited_node_chosen",
      "choose_node1",
      "observations_before_stop",
      "stop_decision_timestep",
      "timestep_before_stop",
      "kl_paid_total",
      "kl_paid_per_stop_timestep"
    )
  )
  formula <- as.formula(paste("proportion_timesteps_node1 ~", paste(group_cols, collapse = " + ")))
  count_data <- aggregate(
    formula,
    data = trial_metrics,
    FUN = length
  )
  names(count_data)[names(count_data) == "proportion_timesteps_node1"] <- "n"
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
  color_mode = "parameter"
)
plot_summary_scatter(
  average_summary,
  x_col = "normalized_chosen_path_reward",
  y_col = "timestep_before_stop",
  xlab = "Average normalized chosen path reward",
  ylab = "Average timestep before stopping",
  file_prefix = "revisit_average_timestep_before_stop_vs_average_normalized_chosen_path_reward_by_sigma",
  facet_sigma = TRUE,
  color_mode = "parameter"
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
  color_mode = "parameter"
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
  color_mode = "sigma"
)
plot_summary_scatter(
  average_summary,
  x_col = "normalized_chosen_path_reward",
  y_col = "timestep_before_stop",
  xlab = "Average normalized chosen path reward",
  ylab = "Average timestep before stopping",
  file_prefix = "revisit_average_timestep_before_stop_vs_average_normalized_chosen_path_reward_sigma_color",
  facet_sigma = FALSE,
  color_mode = "sigma"
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
  color_mode = "sigma"
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
    color_mode = "parameter"
  )
  plot_summary_scatter(
    average_summary,
    x_col = "timestep_before_stop",
    y_col = entropy_summary_col,
    xlab = "Average timestep before stopping",
    ylab = entropy_summary_label,
    file_prefix = "revisit_average_terminal_choice_entropy_vs_average_timestep_before_stop_by_sigma",
    facet_sigma = TRUE,
    color_mode = "parameter"
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
    color_mode = "sigma"
  )
  plot_summary_scatter(
    average_summary,
    x_col = "timestep_before_stop",
    y_col = entropy_summary_col,
    xlab = "Average timestep before stopping",
    ylab = entropy_summary_label,
    file_prefix = "revisit_average_terminal_choice_entropy_vs_average_timestep_before_stop_sigma_color",
    facet_sigma = FALSE,
    color_mode = "sigma"
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
node_difference_summary <- summarize_difference_metrics(
  node_difference_trial_metrics,
  "node_reward_difference"
)
last_visited_difference_summary <- summarize_difference_metrics(
  node_difference_trial_metrics,
  "last_visited_reward_difference"
)
absolute_node_difference_summary <- summarize_difference_metrics(
  node_difference_trial_metrics,
  "absolute_node_reward_difference"
)
plot_metric_by_difference(
  node_difference_summary,
  y_col = "proportion_timesteps_node1",
  ylab = "Proportion of pre-stop timesteps visiting node 1",
  file_prefix = "revisit_node1_minus_node2_reward_vs_proportion_node1_visits_by_sigma"
)
plot_metric_by_difference(
  last_visited_difference_summary,
  y_col = "last_visited_node_chosen",
  ylab = "P(last visited node is chosen)",
  file_prefix = "revisit_last_visited_minus_other_reward_vs_last_visited_node_chosen_by_sigma",
  x_col = "last_visited_reward_difference",
  xlab = "Last visited node actual reward - other node actual reward"
)
plot_metric_by_difference(
  node_difference_summary,
  y_col = "choose_node1",
  ylab = "P(choose node 1)",
  file_prefix = "revisit_node1_minus_node2_reward_vs_probability_choose_node1_by_sigma"
)
plot_metric_by_difference(
  absolute_node_difference_summary,
  y_col = "timestep_before_stop",
  ylab = "Average timestep before stopping",
  file_prefix = "revisit_node1_minus_node2_reward_vs_average_timestep_before_stop_by_sigma",
  x_col = "absolute_node_reward_difference",
  xlab = "|node 1 actual reward - node 2 actual reward|"
)
plot_metric_by_difference(
  absolute_node_difference_summary,
  y_col = "kl_paid_total",
  ylab = "Average KL paid across timesteps",
  file_prefix = "revisit_node1_minus_node2_reward_vs_average_kl_paid_by_sigma",
  x_col = "absolute_node_reward_difference",
  xlab = "|node 1 actual reward - node 2 actual reward|"
)
plot_metric_by_difference(
  absolute_node_difference_summary,
  y_col = "kl_paid_per_stop_timestep",
  ylab = "Average KL paid per stop timestep",
  file_prefix = "revisit_node1_minus_node2_reward_vs_average_kl_paid_per_timestep_before_stop_by_sigma",
  x_col = "absolute_node_reward_difference",
  xlab = "|node 1 actual reward - node 2 actual reward|"
)
