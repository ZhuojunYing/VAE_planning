#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)

get_arg <- function(i, default) {
  if (length(args) >= i && nzchar(args[[i]])) args[[i]] else default
}

memory_lambda_arg <- get_arg(1, "0.01,0.1,1.0,2.0,4.0,6.0,8.0,10.0")
loss_scale_arg <- get_arg(2, "1.0")
# Backward-compatible internal names: old beta now means direct memory lambda;
# old lambda now means reward/action loss scale.
beta_arg <- memory_lambda_arg
lambda_arg <- loss_scale_arg
alpha_arg <- get_arg(3, "0.0")
opportunity_arg <- get_arg(4, "0.0")
input_dir <- get_arg(5, "outputs/simulations")
results_dir <- get_arg(6, "results")
tree_size <- as.integer(get_arg(7, "2"))
input_type <- get_arg(8, "binary")
expansion_decision_version <- get_arg(9, "decoder")
model_variant <- get_arg(10, "vae")
tree_config <- get_arg(11, "")
exact_dir <- get_arg(12, "analyses/exp_binary/results/exact_time_cost")
seed_arg <- get_arg(13, "auto")
optimal_time_cost_arg <- get_arg(14, opportunity_arg)
rnn_units_arg <- get_arg(15, "64")
latent_dim_arg <- get_arg(16, "32")
source_arg_values <- c("tf", "tensorflow", "jax")
arg17 <- get_arg(17, "")
if (tolower(trimws(arg17)) %in% source_arg_values) {
  zero_exact_dir <- "analyses/exp_binary/results/exact_time_cost_zero"
  simulation_source_arg <- arg17
  sigma_arg <- get_arg(18, "0")
} else {
  zero_exact_dir <- get_arg(17, "analyses/exp_binary/results/exact_time_cost_zero")
  simulation_source_arg <- get_arg(18, "tensorflow")
  sigma_arg <- get_arg(19, "0")
}

normalize_simulation_source <- function(source) {
  source_key <- tolower(trimws(as.character(source)))
  aliases <- c(
    "tf" = "tensorflow",
    "tensorflow" = "tensorflow",
    "keras" = "tensorflow",
    "jax" = "jax"
  )
  if (!source_key %in% names(aliases)) {
    stop(sprintf(
      "simulation_source must be one of: %s. Got %s.",
      paste(names(aliases), collapse = ", "),
      source
    ))
  }
  unname(aliases[[source_key]])
}

simulation_source <- normalize_simulation_source(simulation_source_arg)
if (
  identical(simulation_source, "jax") &&
    normalizePath(input_dir, mustWork = FALSE) == normalizePath("outputs/simulations", mustWork = FALSE)
) {
  input_dir <- "outputs/jax_simulations"
}
simulation_source_suffix <- if (identical(simulation_source, "jax")) "_source_jax" else ""

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

expansion_decision_version <- normalize_expansion_decision_version(expansion_decision_version)
model_variant <- normalize_model_variant(model_variant)
tree_config <- normalize_tree_config(tree_config)
tree_file_label <- paste0(tree_size, "n", if (nzchar(tree_config)) paste0("_", tree_config) else "")
architecture_file_label <- sprintf("rnn_%s_latent_%s", rnn_units_arg, latent_dim_arg)

dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)
message(sprintf("Using %s simulation CSVs from %s", simulation_source, input_dir))

panel_width_in <- 60 / 25.4
panel_height_in <- 60 / 25.4
plot_font_size_pt <- 7
legend_panel_fraction <- 0.55

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
  par(
    cex = 1,
    cex.axis = 1,
    cex.lab = 1,
    cex.main = 1,
    cex.sub = 1
  )
}

beta_values <- trimws(strsplit(beta_arg, ",")[[1]])
opportunity_values <- trimws(strsplit(opportunity_arg, ",")[[1]])
optimal_time_cost_values <- trimws(strsplit(optimal_time_cost_arg, ",")[[1]])
sigma_values <- trimws(strsplit(sigma_arg, ",")[[1]])
auto_seeds <- identical(tolower(trimws(seed_arg)), "auto")
seed_values <- if (auto_seeds) {
  integer()
} else {
  as.integer(trimws(strsplit(seed_arg, ",")[[1]]))
}

arg_label <- function(values) {
  label <- paste(values, collapse = "_")
  gsub("[^A-Za-z0-9._-]+", "_", label)
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
  value_nums <- suppressWarnings(as.numeric(values))
  if (all(!is.na(value_nums))) {
    return(sprintf(
      "n%d_min_%s_max_%s",
      length(value_nums),
      short_num_label(min(value_nums)),
      short_num_label(max(value_nums))
    ))
  }
  sprintf("n%d_%s", length(values), substr(full_label, 1, max_full_chars))
}

expansion_label <- sprintf(
  "%s_variant_%s_%s",
  arg_label(expansion_decision_version),
  arg_label(model_variant),
  arg_label(architecture_file_label)
)
lambda_label <- compact_arg_label(trimws(strsplit(lambda_arg, ",")[[1]]))
alpha_label <- compact_arg_label(trimws(strsplit(alpha_arg, ",")[[1]]))
opportunity_label <- compact_arg_label(opportunity_values)
beta_label <- compact_arg_label(beta_values)
sigma_label <- compact_arg_label(sigma_values)
optimal_time_cost_label <- compact_arg_label(optimal_time_cost_values)
sigma_numeric_values <- suppressWarnings(as.numeric(sigma_values))
sigma_arg_is_default <- length(sigma_values) == 1L &&
  !is.na(sigma_numeric_values[[1]]) &&
  abs(sigma_numeric_values[[1]]) < 1e-12
sigma_file_suffix <- if (sigma_arg_is_default) "" else paste0("_sigma_", sigma_label)
optimal_time_cost_suffix <- if (!identical(optimal_time_cost_arg, opportunity_arg)) {
  "_optimal"
} else {
  ""
}

model_variant_file_segments <- function(variant) {
  segments <- sprintf("variant_%s_", variant)
  if (identical(variant, "vae")) {
    segments <- c(segments, "")
  }
  unique(segments)
}

variant_file_segments <- model_variant_file_segments(model_variant)

simulation_tree_file_labels <- function() {
  labels <- paste0(tree_file_label, "_", architecture_file_label)
  if (identical(as.character(rnn_units_arg), "64") && identical(as.character(latent_dim_arg), "32")) {
    labels <- c(labels, tree_file_label)
  }
  unique(labels)
}

simulation_optional_suffix_regex <- function() {
  # JAX filenames may include non-default analysis/training suffixes before
  # the input-type tag, e.g. _obs_sigma_0.5_klstart_5_klanneal_60.
  # Do not match revisit suffixes here; those runs are intentionally separate.
  "(_obs_sigma_[^_]+)?(_klstart_[^_]+_klanneal_[0-9]+)?(_stop_paid)?(_observer_endchoice)?"
}

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
    header <- names(data.table::fread(
      path,
      nrows = 0,
      data.table = FALSE,
      showProgress = FALSE
    ))
    keep <- header[header %in% keep_names]
    if (length(keep_patterns) > 0) {
      keep <- unique(c(
        keep,
        header[vapply(header, function(col) {
          any(grepl(paste(keep_patterns, collapse = "|"), col))
        }, logical(1))]
      ))
    }
    if (length(keep) > 0) {
      select_cols <- keep
    }
    dat <- data.table::fread(
      path,
      select = select_cols,
      data.table = FALSE,
      showProgress = FALSE
    )
    return(drop_unnamed_index_columns(as.data.frame(dat)))
  }

  dat <- read.csv(path, stringsAsFactors = FALSE)
  dat <- drop_unnamed_index_columns(dat)
  if (length(keep_names) == 0 && length(keep_patterns) == 0) {
    return(dat)
  }
  keep <- names(dat)[names(dat) %in% keep_names]
  if (length(keep_patterns) > 0) {
    keep <- unique(c(
      keep,
      names(dat)[vapply(names(dat), function(col) {
        any(grepl(paste(keep_patterns, collapse = "|"), col))
      }, logical(1))]
    ))
  }
  dat[, keep, drop = FALSE]
}

simulation_keep_names <- c(
  "V",
  "graph",
  "node",
  "actual_reward",
  "chosen_path",
  "opportunity_cost",
  "observation_sigma",
  "expansion_decision_version"
)
simulation_keep_patterns <- c(
  "^expanded_reward_t[0-9]+$",
  "^expanded_node_t[0-9]+$",
  "^stop_t[0-9]+$",
  "^kl_d_t[0-9]+$"
)

value_candidates <- function(x) {
  x_chr <- as.character(x)
  x_num <- suppressWarnings(as.numeric(x_chr))
  candidates <- x_chr
  if (!is.na(x_num)) {
    rounded_1 <- round(x_num, 1)
    rounded_2 <- round(x_num, 2)
    candidates <- c(candidates, format(x_num, scientific = FALSE, trim = TRUE))
    if (abs(x_num - rounded_1) < 1e-12) {
      candidates <- c(candidates, sprintf("%.1f", rounded_1))
    }
    if (abs(x_num - rounded_2) < 1e-12) {
      candidates <- c(candidates, sprintf("%.2f", rounded_2))
    }
  }
  unique(candidates)
}

numeric_file_match <- function(lambda_value, alpha_value, beta_value, opportunity_value, seed, sigma_value) {
  requested <- suppressWarnings(as.numeric(c(
    lambda_value, alpha_value, beta_value, opportunity_value
  )))
  if (any(is.na(requested))) {
    return(NA_character_)
  }

  files <- list.files(input_dir, full.names = TRUE)
  for (tree_label_candidate in simulation_tree_file_labels()) {
    for (variant_file_segment in variant_file_segments) {
      patterns <- c(
        paste0(
          "^loss_scale_([^_]+)_alpha_([^_]+)_lambda_([^_]+)_opportunity_([^_]+)_",
          "expansion_", expansion_decision_version, "_", variant_file_segment,
          "seed_", seed, "_", tree_label_candidate,
          simulation_optional_suffix_regex(),
          "_", input_type, "\\.csv$"
        ),
        paste0(
          "^lambda_([^_]+)_alpha_([^_]+)_beta_([^_]+)_opportunity_([^_]+)_",
          "expansion_", expansion_decision_version, "_", variant_file_segment,
          "seed_", seed, "_", tree_label_candidate,
          simulation_optional_suffix_regex(),
          "_", input_type, "\\.csv$"
        )
      )
      for (pattern in patterns) {
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
            if (sigma_matches(files[[i]], sigma_value)) {
              return(files[[i]])
            }
          }
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
                  file_names <- c(
                    sprintf(
                      "loss_scale_%s_alpha_%s_lambda_%s_opportunity_%s_expansion_%s_%sseed_%d_%s%s_%s.csv",
                      lambda_candidate, alpha_candidate, beta_candidate, opportunity_candidate,
                      expansion_decision_version, variant_file_segment, seed, tree_label_candidate, suffix, input_type
                    ),
                    sprintf(
                      "lambda_%s_alpha_%s_beta_%s_opportunity_%s_expansion_%s_%sseed_%d_%s%s_%s.csv",
                      lambda_candidate, alpha_candidate, beta_candidate, opportunity_candidate,
                      expansion_decision_version, variant_file_segment, seed, tree_label_candidate, suffix, input_type
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
      }
    }
  }
  numeric_file_match(lambda_value, alpha_value, beta_value, opportunity_value, seed, sigma_value)
}

matching_simulation_files <- function(lambda_value, alpha_value, beta_value, opportunity_value, sigma_value) {
  requested <- suppressWarnings(as.numeric(c(
    lambda_value, alpha_value, beta_value, opportunity_value
  )))
  if (any(is.na(requested))) {
    return(data.frame(path = character(), seed = integer(), stringsAsFactors = FALSE))
  }

  files <- list.files(input_dir, full.names = TRUE)
  rows <- list()
  for (tree_label_candidate in simulation_tree_file_labels()) {
    tree_rows <- list()
    for (variant_file_segment in variant_file_segments) {
      patterns <- c(
        paste0(
          "^loss_scale_([^_]+)_alpha_([^_]+)_lambda_([^_]+)_opportunity_([^_]+)_",
          "expansion_", expansion_decision_version, "_", variant_file_segment,
          "seed_([0-9]+)_", tree_label_candidate,
          simulation_optional_suffix_regex(),
          "_", input_type, "\\.csv$"
        ),
        paste0(
          "^lambda_([^_]+)_alpha_([^_]+)_beta_([^_]+)_opportunity_([^_]+)_",
          "expansion_", expansion_decision_version, "_", variant_file_segment,
          "seed_([0-9]+)_", tree_label_candidate,
          simulation_optional_suffix_regex(),
          "_", input_type, "\\.csv$"
        )
      )
      for (pattern in patterns) {
        matches <- regexec(pattern, basename(files))
        pieces <- regmatches(basename(files), matches)
        for (i in seq_along(pieces)) {
          if (length(pieces[[i]]) == 0) {
            next
          }
          found <- suppressWarnings(as.numeric(pieces[[i]][2:5]))
          seed_value <- suppressWarnings(as.integer(pieces[[i]][6]))
          if (any(is.na(found)) || is.na(seed_value)) {
            next
          }
          if (all(abs(found - requested) < 1e-8) && sigma_matches(files[[i]], sigma_value)) {
            tree_rows[[length(tree_rows) + 1]] <- data.frame(
              path = files[[i]],
              seed = seed_value,
              sigma = sigma_value,
              stringsAsFactors = FALSE
            )
          }
        }
      }
    }
    if (length(tree_rows) > 0) {
      rows <- tree_rows
      break
    }
  }

  if (length(rows) == 0) {
    return(data.frame(path = character(), seed = integer(), stringsAsFactors = FALSE))
  }
  out <- unique(do.call(rbind, rows))
  if (!auto_seeds) {
    out <- out[out$seed %in% seed_values, , drop = FALSE]
  }
  out[order(out$seed, out$path), , drop = FALSE]
}

read_seed_file <- function(beta_value, opportunity_value, seed, sigma_value) {
  file_path <- simulation_path(lambda_arg, alpha_arg, beta_value, opportunity_value, seed, sigma_value)
  if (is.na(file_path)) {
    warning(sprintf(
      "Missing simulation file for memory lambda=%s opportunity=%s sigma=%s seed=%d model_variant=%s",
      beta_value, opportunity_value, sigma_value, seed, model_variant
    ))
    return(NULL)
  }
  dat <- read_csv_fast(
    file_path,
    keep_names = simulation_keep_names,
    keep_patterns = simulation_keep_patterns
  )
  dat <- unique(dat)
  dat$beta <- beta_value
  dat$opportunity <- opportunity_value
  dat$sigma <- sigma_value
  dat$seed <- seed
  dat$model_variant <- model_variant
  dat$file_path <- file_path
  dat$source_file <- file_path
  dat
}

read_simulation_file <- function(file_path, beta_value, opportunity_value, seed, sigma_value) {
  dat <- read_csv_fast(
    file_path,
    keep_names = simulation_keep_names,
    keep_patterns = simulation_keep_patterns
  )
  dat <- unique(dat)
  dat$beta <- beta_value
  dat$opportunity <- opportunity_value
  dat$sigma <- sigma_value
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
    for (sigma_value in sigma_values) {
      if (auto_seeds) {
        combo_files <- matching_simulation_files(
          lambda_arg,
          alpha_arg,
          beta_value,
          opportunity_value,
          sigma_value
        )
        if (nrow(combo_files) == 0) {
          warning(sprintf(
            "Missing simulation files for memory lambda=%s opportunity=%s sigma=%s model_variant=%s",
            beta_value, opportunity_value, sigma_value, model_variant
          ))
        }
        for (file_i in seq_len(nrow(combo_files))) {
          seed_data <- read_simulation_file(
            combo_files$path[[file_i]],
            beta_value,
            opportunity_value,
            combo_files$seed[[file_i]],
            combo_files$sigma[[file_i]]
          )
          loaded_data[[length(loaded_data) + 1]] <- seed_data
        }
      } else {
        for (seed in seed_values) {
          seed_data <- read_seed_file(beta_value, opportunity_value, seed, sigma_value)
          if (!is.null(seed_data)) {
            loaded_data[[length(loaded_data) + 1]] <- seed_data
          }
        }
      }
    }
  }
}

all_data <- bind_rows_fill(loaded_data)
if (is.null(all_data) || nrow(all_data) == 0) {
  stop("No simulation CSVs were found. Check memory-lambda/loss-scale/alpha values and input_dir.")
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
}

if ("expansion_decision_version" %in% names(all_data)) {
  all_data <- all_data[
    as.character(all_data$expansion_decision_version) == expansion_decision_version,
    ,
    drop = FALSE
  ]
}

if (nrow(all_data) == 0) {
  stop("No rows remained after filtering opportunity_cost and expansion_decision_version.")
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

unique_trial_rows <- function(dat, required_cols) {
  trial_id_cols <- intersect(c("sigma", "beta", "opportunity", "seed", "graph"), names(dat))
  trial_cols <- unique(c(trial_id_cols, required_cols))
  unique(dat[, trial_cols, drop = FALSE])
}

summarize_probability <- function(dat, group_cols, action_col, out_col) {
  if (nrow(dat) == 0) {
    out <- data.frame(stringsAsFactors = FALSE)
    for (col in group_cols) {
      out[[col]] <- dat[[col]][FALSE]
    }
    out[[out_col]] <- numeric()
    out$n <- integer()
    return(out[, c(group_cols, out_col, "n"), drop = FALSE])
  }
  summary_data <- aggregate(dat[[action_col]], dat[, group_cols, drop = FALSE], FUN = mean)
  names(summary_data)[ncol(summary_data)] <- out_col
  count_data <- aggregate(dat[[action_col]], dat[, group_cols, drop = FALSE], FUN = length)
  names(count_data)[ncol(count_data)] <- "n"
  merge(summary_data, count_data, by = group_cols)
}

row_alive_before_decision <- function(trial_data, decision_timestep) {
  alive <- rep(TRUE, nrow(trial_data))
  if (decision_timestep <= 1L) {
    return(alive)
  }
  for (previous_timestep in seq_len(decision_timestep - 1L)) {
    stop_col <- paste0("stop_t", previous_timestep)
    if (!stop_col %in% names(trial_data)) {
      next
    }
    stop_values <- trial_data[[stop_col]]
    stop_values[is.na(stop_values)] <- FALSE
    alive <- alive & !as_logical_col(stop_values)
  }
  alive
}

build_difference_continue_feature_summary_fast <- function(dat, max_decision_timestep) {
  if (task_path_count != 2L || max_decision_timestep < 2L) {
    return(NULL)
  }
  required_cols <- unique(c(
    paste0("expanded_node_t", seq_len(max_decision_timestep - 1L)),
    paste0("expanded_reward_t", seq_len(max_decision_timestep - 1L)),
    paste0("stop_t", seq_len(max_decision_timestep))
  ))
  required_cols <- required_cols[required_cols %in% names(dat)]
  trial_data <- unique_trial_rows(dat, required_cols)
  actual_lookup <- build_node_actual_reward_lookup(dat)
  rows <- list()
  row_index <- seq_len(nrow(trial_data))

  for (decision_timestep in 2:max_decision_timestep) {
    stop_col <- paste0("stop_t", decision_timestep)
    previous_node_col <- paste0("expanded_node_t", decision_timestep - 1L)
    if (!all(c(stop_col, previous_node_col) %in% names(trial_data))) {
      next
    }

    path_values <- matrix(0, nrow = nrow(trial_data), ncol = 2L)
    for (observed_timestep in seq_len(decision_timestep - 1L)) {
      node_col <- paste0("expanded_node_t", observed_timestep)
      reward_col <- paste0("expanded_reward_t", observed_timestep)
      if (!all(c(node_col, reward_col) %in% names(trial_data))) {
        next
      }
      node_values <- suppressWarnings(as.integer(trial_data[[node_col]]))
      reward_values_t <- observed_actual_reward_values(trial_data, node_col, reward_col, actual_lookup)
      path_indices <- path_id_for_node(node_values)
      valid_obs <- !is.na(path_indices) &
        !is.na(reward_values_t) &
        path_indices >= 1L &
        path_indices <= 2L
      if (any(valid_obs)) {
        idx <- cbind(row_index[valid_obs], path_indices[valid_obs])
        path_values[idx] <- path_values[idx] + reward_values_t[valid_obs]
      }
    }

    current_path <- path_id_for_node(trial_data[[previous_node_col]])
    valid_current <- !is.na(current_path) & current_path >= 1L & current_path <= 2L
    other_path <- ifelse(current_path == 1L, 2L, 1L)
    path_difference <- rep(NA_real_, nrow(trial_data))
    path_difference[valid_current] <- path_values[cbind(row_index[valid_current], current_path[valid_current])] -
      path_values[cbind(row_index[valid_current], other_path[valid_current])]

    stop_raw <- trial_data[[stop_col]]
    valid_stop <- !is.na(stop_raw)
    alive <- row_alive_before_decision(trial_data, decision_timestep)
    keep <- valid_stop & alive & valid_current & is.finite(path_difference)
    if (!any(keep)) {
      next
    }

    rows[[length(rows) + 1L]] <- data.frame(
      sigma = trial_data$sigma[keep],
      beta = trial_data$beta[keep],
      opportunity = trial_data$opportunity[keep],
      decision_timestep = decision_timestep,
      path_value_difference = path_difference[keep],
      continue_action = as.numeric(!as_logical_col(stop_raw[keep])),
      stringsAsFactors = FALSE
    )
  }

  if (length(rows) == 0) {
    return(data.frame())
  }
  continue_data <- do.call(rbind, rows)
  summarize_probability(
    continue_data,
    group_cols = c("sigma", "beta", "opportunity", "decision_timestep", "path_value_difference"),
    action_col = "continue_action",
    out_col = "p_continue"
  )
}

build_best_continue_feature_summary_fast <- function(dat, max_decision_timestep) {
  if (max_decision_timestep < 2L) {
    return(data.frame())
  }
  required_cols <- unique(c(
    paste0("expanded_node_t", seq_len(max_decision_timestep - 1L)),
    paste0("expanded_reward_t", seq_len(max_decision_timestep - 1L)),
    paste0("stop_t", seq_len(max_decision_timestep))
  ))
  required_cols <- required_cols[required_cols %in% names(dat)]
  trial_data <- unique_trial_rows(dat, required_cols)
  if (nrow(trial_data) == 0) {
    return(data.frame())
  }
  actual_lookup <- build_node_actual_reward_lookup(dat)

  rows <- list()
  row_index <- seq_len(nrow(trial_data))
  path_values <- matrix(0, nrow = nrow(trial_data), ncol = task_path_count)
  path_counts <- matrix(0L, nrow = nrow(trial_data), ncol = task_path_count)

  for (decision_timestep in 2:max_decision_timestep) {
    observed_timestep <- decision_timestep - 1L
    node_col <- paste0("expanded_node_t", observed_timestep)
    reward_col <- paste0("expanded_reward_t", observed_timestep)
    if (all(c(node_col, reward_col) %in% names(trial_data))) {
      node_values <- suppressWarnings(as.integer(trial_data[[node_col]]))
      reward_values_t <- observed_actual_reward_values(trial_data, node_col, reward_col, actual_lookup)
      path_indices <- path_id_for_node(node_values)
      valid_obs <- !is.na(path_indices) &
        !is.na(reward_values_t) &
        path_indices >= 1L &
        path_indices <= task_path_count
      if (any(valid_obs)) {
        idx <- cbind(row_index[valid_obs], path_indices[valid_obs])
        path_values[idx] <- path_values[idx] + reward_values_t[valid_obs]
        path_counts[idx] <- path_counts[idx] + 1L
      }
    }

    stop_col <- paste0("stop_t", decision_timestep)
    if (!stop_col %in% names(trial_data)) {
      next
    }
    observed_path_values <- path_values
    observed_path_values[path_counts <= 0L] <- -Inf
    best_path_values <- do.call(
      pmax,
      c(as.data.frame(observed_path_values, check.names = FALSE), list(na.rm = FALSE))
    )
    stop_raw <- trial_data[[stop_col]]
    valid_stop <- !is.na(stop_raw)
    alive <- row_alive_before_decision(trial_data, decision_timestep)
    observed_any <- rowSums(path_counts > 0L) > 0L
    keep <- valid_stop & alive & observed_any & is.finite(best_path_values)
    if (!any(keep)) {
      next
    }

    row_data <- data.frame(
      sigma = trial_data$sigma[keep],
      beta = trial_data$beta[keep],
      opportunity = trial_data$opportunity[keep],
      decision_timestep = decision_timestep,
      best_path_value = best_path_values[keep],
      continue_action = as.numeric(!as_logical_col(stop_raw[keep])),
      stringsAsFactors = FALSE
    )
    if (is_disjoint3x2) {
      best_mask <- path_counts > 0L & abs(path_values - best_path_values) < 1e-12
      complete_mask <- path_counts >= task_nodes_per_path
      row_data$best_path_complete <- ifelse(
        rowSums(best_mask[keep, , drop = FALSE] & complete_mask[keep, , drop = FALSE]) > 0L,
        "complete",
        "incomplete"
      )
    }
    rows[[length(rows) + 1L]] <- row_data
  }

  if (length(rows) == 0) {
    return(data.frame())
  }
  continue_data <- do.call(rbind, rows)
  group_cols <- c("sigma", "beta", "opportunity", "decision_timestep", "best_path_value")
  if (is_disjoint3x2 && "best_path_complete" %in% names(continue_data)) {
    continue_data <- continue_data[!is.na(continue_data$best_path_complete), , drop = FALSE]
    group_cols <- c(group_cols, "best_path_complete")
  }
  summarize_probability(
    continue_data,
    group_cols = group_cols,
    action_col = "continue_action",
    out_col = "p_continue"
  )
}

is_bandit3 <- identical(tree_config, "bandit3") || tree_size == 3
is_bandit4 <- identical(tree_config, "bandit4")
is_default2 <- tree_size == 2 && !nzchar(tree_config)
is_disjoint2x2 <- identical(tree_config, "disjoint2x2")
is_disjoint3x2 <- identical(tree_config, "disjoint3x2")

task_name <- if (is_bandit3) {
  "bandit3"
} else if (is_bandit4) {
  "bandit4"
} else if (is_disjoint2x2) {
  "disjoint2x2"
} else if (is_disjoint3x2) {
  "disjoint3x2"
} else if (is_default2) {
  "default2"
} else {
  ""
}

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

task_path_nodes <- function() {
  split(seq_len(task_node_count), rep(seq_len(task_path_count), each = task_nodes_per_path))
}

expected_best_path_reward <- function() {
  grids <- expand.grid(rep(list(reward_values), task_node_count))
  path_nodes <- task_path_nodes()
  path_rewards <- vapply(path_nodes, function(nodes) {
    rowSums(grids[, nodes, drop = FALSE])
  }, numeric(nrow(grids)))
  mean(apply(path_rewards, 1, max))
}

task_reward_norm <- expected_best_path_reward()

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

path_state_before_decision <- function(row_data, decision_timestep, actual_lookup = NULL) {
  values <- rep(0, task_path_count)
  counts <- rep(0L, task_path_count)
  if (decision_timestep <= 1) {
    return(list(values = values, counts = counts))
  }
  for (observed_timestep in seq_len(decision_timestep - 1L)) {
    node_col <- paste0("expanded_node_t", observed_timestep)
    reward_col <- paste0("expanded_reward_t", observed_timestep)
    if (!node_col %in% names(row_data) || !reward_col %in% names(row_data)) {
      next
    }
    node_value <- suppressWarnings(as.integer(row_data[[node_col]][[1]]))
    reward_value <- observed_actual_reward_values(row_data, node_col, reward_col, actual_lookup)[[1]]
    path_index <- path_id_for_node(node_value)
    if (
      !is.na(path_index) && !is.na(reward_value) &&
        path_index >= 1L && path_index <= task_path_count
    ) {
      values[[path_index]] <- values[[path_index]] + reward_value
      counts[[path_index]] <- counts[[path_index]] + 1L
    }
  }
  list(values = values, counts = counts)
}

path_state_after_observation <- function(row_data, observation_timestep, actual_lookup = NULL) {
  values <- rep(0, task_path_count)
  counts <- rep(0L, task_path_count)
  if (observation_timestep < 1) {
    return(list(values = values, counts = counts))
  }
  for (observed_timestep in seq_len(observation_timestep)) {
    node_col <- paste0("expanded_node_t", observed_timestep)
    reward_col <- paste0("expanded_reward_t", observed_timestep)
    if (!node_col %in% names(row_data) || !reward_col %in% names(row_data)) {
      next
    }
    node_value <- suppressWarnings(as.integer(row_data[[node_col]][[1]]))
    reward_value <- observed_actual_reward_values(row_data, node_col, reward_col, actual_lookup)[[1]]
    path_index <- path_id_for_node(node_value)
    if (
      !is.na(path_index) && !is.na(reward_value) &&
        path_index >= 1L && path_index <= task_path_count
    ) {
      values[[path_index]] <- values[[path_index]] + reward_value
      counts[[path_index]] <- counts[[path_index]] + 1L
    }
  }
  list(values = values, counts = counts)
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

build_trial_path_reward_lookup <- function(dat) {
  key_cols <- intersect(c("sigma", "beta", "opportunity", "seed", "graph"), names(dat))
  if (!all(c("node", "actual_reward") %in% names(dat))) {
    return(list(key_cols = key_cols, rewards = list()))
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
    return(list(key_cols = key_cols, rewards = list()))
  }
  node_data$key <- trial_key_values(node_data, key_cols)
  split_rows <- split(node_data, node_data$key)
  path_nodes <- split(seq_len(task_node_count), rep(seq_len(task_path_count), each = task_nodes_per_path))
  reward_lookup <- lapply(split_rows, function(piece) {
    node_rewards <- rep(NA_real_, task_node_count)
    for (node_i in seq_len(nrow(piece))) {
      node_rewards[[piece$node[[node_i]]]] <- piece$actual_reward[[node_i]]
    }
    vapply(path_nodes, function(nodes) sum(node_rewards[nodes]), numeric(1))
  })
  list(key_cols = key_cols, rewards = reward_lookup)
}

normalize_chosen_path_reward <- function(chosen_path_reward, path_rewards) {
  if (is.na(chosen_path_reward) || !is.finite(task_reward_norm) || abs(task_reward_norm) < 1e-12) {
    return(NA_real_)
  }
  chosen_path_reward / task_reward_norm
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

build_trial_diagnostics <- function(dat) {
  reward_timesteps <- column_timesteps(dat, "^expanded_reward_t[0-9]+$", "^expanded_reward_t")
  stop_timesteps <- column_timesteps(dat, "^stop_t[0-9]+$", "^stop_t")
  kl_timesteps <- column_timesteps(dat, "^kl_d_t[0-9]+$", "^kl_d_t")
  required_cols <- unique(c(
    "V",
    paste0("expanded_reward_t", reward_timesteps),
    paste0("stop_t", stop_timesteps),
    paste0("kl_d_t", kl_timesteps)
  ))
  required_cols <- required_cols[required_cols %in% names(dat)]
  trial_data <- unique_trial_rows(dat, required_cols)
  if (!"V" %in% names(trial_data)) {
    stop("Simulation rows must include V, the actual reward of the chosen path.")
  }

  trial_data$chosen_path_reward <- suppressWarnings(as.numeric(trial_data$V))
  trial_data$normalized_chosen_path_reward <- trial_data$chosen_path_reward / task_reward_norm

  kl_cols <- paste0("kl_d_t", kl_timesteps)
  kl_cols <- kl_cols[kl_cols %in% names(trial_data)]
  kl_mat <- numeric_matrix_from_cols(trial_data, kl_cols)
  kl_mat[!is.finite(kl_mat)] <- 0
  trial_data$kl_paid_total <- rowSums(kl_mat)

  reward_cols <- paste0("expanded_reward_t", reward_timesteps)
  reward_cols <- reward_cols[reward_cols %in% names(trial_data)]
  reward_mat <- numeric_matrix_from_cols(trial_data, reward_cols)
  trial_data$observations_before_stop <- rowSums(is.finite(reward_mat))
  trial_data[
    !is.na(trial_data$chosen_path_reward),
    ,
    drop = FALSE
  ]
}

build_continue_feature_summary <- function(dat, feature_type) {
  max_decision_timestep <- min(task_node_count, tree_size)
  if (max_decision_timestep < 2) {
    return(data.frame())
  }
  if (identical(feature_type, "best")) {
    fast_summary <- build_best_continue_feature_summary_fast(dat, max_decision_timestep)
    if (!is.null(fast_summary)) {
      return(fast_summary)
    }
  }
  if (!identical(feature_type, "best") && task_path_count == 2L) {
    fast_summary <- build_difference_continue_feature_summary_fast(dat, max_decision_timestep)
    if (!is.null(fast_summary)) {
      return(fast_summary)
    }
  }
  required_cols <- unique(c(
    paste0("expanded_node_t", seq_len(max_decision_timestep - 1L)),
    paste0("expanded_reward_t", seq_len(max_decision_timestep - 1L)),
    paste0("stop_t", seq_len(max_decision_timestep))
  ))
  required_cols <- required_cols[required_cols %in% names(dat)]
  trial_data <- unique_trial_rows(dat, required_cols)
  actual_lookup <- build_node_actual_reward_lookup(dat)
  rows <- list()

  for (decision_timestep in 2:max_decision_timestep) {
    stop_col <- paste0("stop_t", decision_timestep)
    if (!stop_col %in% names(trial_data)) {
      next
    }
    for (row_i in seq_len(nrow(trial_data))) {
      row_data <- trial_data[row_i, , drop = FALSE]
      if (is.na(row_data[[stop_col]][[1]]) || !trial_alive_before_decision(row_data, decision_timestep)) {
        next
      }
      state <- path_state_before_decision(row_data, decision_timestep, actual_lookup)
      if (feature_type == "best") {
        observed_values <- state$values[state$counts > 0L]
        if (length(observed_values) == 0) {
          next
        }
        feature_value <- max(observed_values)
        feature_col <- "best_path_value"
        best_path_complete <- if (is_disjoint3x2) {
          best_indices <- which(state$counts > 0L & abs(state$values - feature_value) < 1e-12)
          if (any(state$counts[best_indices] >= task_nodes_per_path)) "complete" else "incomplete"
        } else {
          NA_character_
        }
      } else {
        if (length(state$values) != 2L) {
          next
        }
        current_node_col <- paste0("expanded_node_t", decision_timestep - 1L)
        if (!current_node_col %in% names(row_data)) {
          next
        }
        current_path <- path_id_for_node(row_data[[current_node_col]][[1]])
        if (is.na(current_path) || current_path < 1L || current_path > 2L) {
          next
        }
        other_path <- if (current_path == 1L) 2L else 1L
        feature_value <- state$values[[current_path]] - state$values[[other_path]]
        feature_col <- "path_value_difference"
        best_path_complete <- NA_character_
      }
      rows[[length(rows) + 1]] <- data.frame(
        sigma = trial_data$sigma[[row_i]],
        beta = trial_data$beta[[row_i]],
        opportunity = trial_data$opportunity[[row_i]],
        decision_timestep = decision_timestep,
        feature_value = feature_value,
        best_path_complete = best_path_complete,
        continue_action = as.numeric(!as_logical_col(row_data[[stop_col]][[1]])),
        stringsAsFactors = FALSE
      )
      names(rows[[length(rows)]])[names(rows[[length(rows)]]) == "feature_value"] <- feature_col
    }
  }

  if (length(rows) == 0) {
    return(data.frame())
  }
  continue_data <- do.call(rbind, rows)
  feature_col <- if (feature_type == "best") "best_path_value" else "path_value_difference"
  group_cols <- c("sigma", "beta", "opportunity", "decision_timestep", feature_col)
  if (feature_type == "best" && is_disjoint3x2 && "best_path_complete" %in% names(continue_data)) {
    continue_data <- continue_data[!is.na(continue_data$best_path_complete), , drop = FALSE]
    group_cols <- c(group_cols, "best_path_complete")
  }
  summarize_probability(
    continue_data,
    group_cols = group_cols,
    action_col = "continue_action",
    out_col = "p_continue"
  )
}

disjoint3x2_condition_from_state <- function(path_values, path_counts) {
  path_values <- as.numeric(path_values)
  path_counts <- as.integer(path_counts)
  path_order <- order(-path_counts, seq_along(path_counts))
  ordered_counts <- path_counts[path_order]
  ordered_values <- path_values[path_order]
  best_value <- max(ordered_values, na.rm = TRUE)
  best_indices <- which(abs(ordered_values - best_value) < 1e-12)
  count_pattern <- paste0("(", paste(ordered_counts, collapse = ","), ")")
  best_path_identity <- if (identical(count_pattern, "(1,0,0)") && !1L %in% best_indices) {
    "not_P1"
  } else if (length(best_indices) == 1L) {
    paste0("P", best_indices[[1]])
  } else {
    "tie"
  }
  list(
    count_pattern = count_pattern,
    best_path_identity = best_path_identity,
    best_path_value = best_value,
    path_order = path_order
  )
}

disjoint3x2_rank_for_path <- function(path_index, path_order) {
  if (is.na(path_index)) {
    return(NA_integer_)
  }
  rank <- match(as.integer(path_index), as.integer(path_order))
  if (is.na(rank)) NA_integer_ else as.integer(rank)
}

build_disjoint3x2_condition_action_summary <- function(dat) {
  if (!is_disjoint3x2 || nrow(dat) == 0) {
    return(data.frame())
  }
  max_decision_timestep <- min(task_node_count, tree_size)
  if (max_decision_timestep < 2L) {
    return(data.frame())
  }
  required_cols <- unique(c(
    "chosen_path",
    paste0("expanded_node_t", seq_len(max_decision_timestep)),
    paste0("expanded_reward_t", seq_len(max_decision_timestep - 1L)),
    paste0("stop_t", seq_len(max_decision_timestep))
  ))
  required_cols <- required_cols[required_cols %in% names(dat)]
  trial_data <- unique_trial_rows(dat, required_cols)
  if (nrow(trial_data) == 0) {
    return(data.frame())
  }
  actual_lookup <- build_node_actual_reward_lookup(dat)

  chosen_path_values <- if ("chosen_path" %in% names(trial_data)) {
    suppressWarnings(as.integer(trial_data$chosen_path))
  } else {
    rep(NA_integer_, nrow(trial_data))
  }
  chosen_path_zero_based <- any(chosen_path_values == 0L, na.rm = TRUE)
  chosen_path_values <- suppressWarnings(as.integer(chosen_path_values))
  if (chosen_path_zero_based) {
    chosen_path_values <- chosen_path_values + 1L
  }
  chosen_path_values[
    is.na(chosen_path_values) |
      chosen_path_values < 1L |
      chosen_path_values > task_path_count
  ] <- NA_integer_

  rows <- list()
  n_trials <- nrow(trial_data)
  row_index <- seq_len(n_trials)
  path_values <- matrix(0, nrow = n_trials, ncol = task_path_count)
  path_counts <- matrix(0L, nrow = n_trials, ncol = task_path_count)
  alive <- rep(TRUE, n_trials)

  canonical_ranks <- function(counts) {
    rank_mat <- matrix(1L, nrow = nrow(counts), ncol = ncol(counts))
    for (path_i in seq_len(ncol(counts))) {
      rank_i <- rep(1L, nrow(counts))
      for (path_j in seq_len(ncol(counts))) {
        if (path_i == path_j) {
          next
        }
        rank_i <- rank_i + as.integer(
          counts[, path_j] > counts[, path_i] |
            (counts[, path_j] == counts[, path_i] & path_j < path_i)
        )
      }
      rank_mat[, path_i] <- rank_i
    }
    rank_mat
  }

  for (decision_timestep in 2:max_decision_timestep) {
    observed_timestep <- decision_timestep - 1L
    node_col <- paste0("expanded_node_t", observed_timestep)
    reward_col <- paste0("expanded_reward_t", observed_timestep)
    if (all(c(node_col, reward_col) %in% names(trial_data))) {
      node_values <- suppressWarnings(as.integer(trial_data[[node_col]]))
      reward_values_t <- observed_actual_reward_values(trial_data, node_col, reward_col, actual_lookup)
      path_indices <- path_id_for_node(node_values)
      valid_obs <- !is.na(path_indices) &
        !is.na(reward_values_t) &
        path_indices >= 1L &
        path_indices <= task_path_count
      if (any(valid_obs)) {
        idx <- cbind(row_index[valid_obs], path_indices[valid_obs])
        path_values[idx] <- path_values[idx] + reward_values_t[valid_obs]
        path_counts[idx] <- path_counts[idx] + 1L
      }
    }

    previous_stop_col <- paste0("stop_t", decision_timestep - 1L)
    if (previous_stop_col %in% names(trial_data)) {
      previous_stop <- trial_data[[previous_stop_col]]
      previous_stop[is.na(previous_stop)] <- FALSE
      alive <- alive & !as_logical_col(previous_stop)
    }

    stop_col <- paste0("stop_t", decision_timestep)
    if (!stop_col %in% names(trial_data)) {
      next
    }
    stop_raw <- trial_data[[stop_col]]
    keep_base <- alive & !is.na(stop_raw)
    if (!any(keep_base)) {
      next
    }

    rank_mat <- canonical_ranks(path_counts)
    ordered_counts <- matrix(NA_integer_, nrow = n_trials, ncol = task_path_count)
    ordered_values <- matrix(NA_real_, nrow = n_trials, ncol = task_path_count)
    for (path_i in seq_len(task_path_count)) {
      rank_i <- rank_mat[, path_i]
      idx <- cbind(row_index, rank_i)
      ordered_counts[idx] <- path_counts[, path_i]
      ordered_values[idx] <- path_values[, path_i]
    }
    best_path_value <- do.call(pmax, c(as.data.frame(ordered_values), list(na.rm = TRUE)))
    best_mask <- abs(ordered_values - best_path_value) < 1e-12
    best_count <- rowSums(best_mask, na.rm = TRUE)
    best_rank <- max.col(best_mask, ties.method = "first")
    count_pattern <- sprintf(
      "(%d,%d,%d)",
      ordered_counts[, 1],
      ordered_counts[, 2],
      ordered_counts[, 3]
    )
    best_path_identity <- ifelse(
      count_pattern == "(1,0,0)" & !best_mask[, 1],
      "not_P1",
      ifelse(best_count == 1L, paste0("P", best_rank), "tie")
    )

    observe_node_col <- paste0("expanded_node_t", decision_timestep)
    observe_path <- if (observe_node_col %in% names(trial_data)) {
      path_id_for_node(trial_data[[observe_node_col]])
    } else {
      rep(NA_integer_, n_trials)
    }
    is_stop <- as_logical_col(stop_raw)
    action_label <- rep(NA_character_, n_trials)
    valid_stop_path <- is_stop & !is.na(chosen_path_values)
    if (any(valid_stop_path)) {
      stop_rank <- rank_mat[cbind(row_index[valid_stop_path], chosen_path_values[valid_stop_path])]
      action_label[valid_stop_path] <- paste0("stop_P", stop_rank)
    }
    action_label[is_stop & is.na(action_label)] <- "stop"
    valid_observe_path <- !is_stop &
      !is.na(observe_path) &
      observe_path >= 1L &
      observe_path <= task_path_count
    if (any(valid_observe_path)) {
      observe_rank <- rank_mat[cbind(row_index[valid_observe_path], observe_path[valid_observe_path])]
      action_label[valid_observe_path] <- paste0("observe_P", observe_rank)
    }

    keep <- keep_base & !is.na(action_label) & is.finite(best_path_value)
    if (!any(keep)) {
      next
    }
    rows[[length(rows) + 1L]] <- data.frame(
      sigma = trial_data$sigma[keep],
      beta = trial_data$beta[keep],
      opportunity = trial_data$opportunity[keep],
      decision_timestep = decision_timestep,
      count_pattern = count_pattern[keep],
      best_path_identity = best_path_identity[keep],
      best_path_value = best_path_value[keep],
      action_label = action_label[keep],
      stringsAsFactors = FALSE
    )
  }

  if (length(rows) == 0) {
    return(data.frame())
  }
  action_data <- do.call(rbind, rows)
  group_cols <- c(
    "sigma",
    "beta",
    "opportunity",
    "decision_timestep",
    "count_pattern",
    "best_path_identity",
    "best_path_value"
  )
  if (requireNamespace("data.table", quietly = TRUE)) {
    action_dt <- data.table::as.data.table(action_data)
    action_counts <- action_dt[
      ,
      .(n = .N),
      by = c(group_cols, "action_label")
    ]
    total_counts <- action_dt[
      ,
      .(total_n = .N),
      by = group_cols
    ]
    out <- merge(action_counts, total_counts, by = group_cols)
    out[, p_action := n / total_n]
    return(as.data.frame(out[, c(group_cols, "action_label", "p_action", "n"), with = FALSE]))
  }
  action_counts <- aggregate(
    rep(1, nrow(action_data)),
    by = action_data[, c(group_cols, "action_label"), drop = FALSE],
    FUN = sum
  )
  names(action_counts)[ncol(action_counts)] <- "n"
  total_counts <- aggregate(
    rep(1, nrow(action_data)),
    by = action_data[, group_cols, drop = FALSE],
    FUN = sum
  )
  names(total_counts)[ncol(total_counts)] <- "total_n"
  out <- merge(action_counts, total_counts, by = group_cols)
  out$p_action <- out$n / out$total_n
  out[, c(group_cols, "action_label", "p_action", "n"), drop = FALSE]
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
  warning("Both memory lambda and opportunity have multiple values; using memory lambda for color and keeping opportunity point/line styles fixed.")
}

color_levels <- if (identical(color_by, "opportunity")) opportunity_levels else beta_levels
color_level_labels <- format_plot_values(color_levels)

color_luminance <- function(cols) {
  rgb_cols <- grDevices::col2rgb(cols) / 255
  0.2126 * rgb_cols[1, ] + 0.7152 * rgb_cols[2, ] + 0.0722 * rgb_cols[3, ]
}

numeric_gradient_colors <- function(levels, palette, darker_high = TRUE) {
  if (length(levels) == 0) {
    return(setNames(character(), character()))
  }
  palette_cols <- grDevices::hcl.colors(max(3, length(levels) + 2), palette = palette)
  palette_cols <- palette_cols[seq_along(levels)]
  light_to_dark <- palette_cols[order(color_luminance(palette_cols), decreasing = TRUE)]
  level_nums <- suppressWarnings(as.numeric(levels))
  level_order <- if (all(!is.na(level_nums))) order(level_nums) else order(as.character(levels))
  assigned <- character(length(levels))
  assigned[level_order] <- if (darker_high) light_to_dark else rev(light_to_dark)
  setNames(assigned, levels)
}

palette_cols <- if (identical(color_by, "opportunity")) {
  # Opportunity cost: blue gradient, darker for higher opportunity cost.
  numeric_gradient_colors(color_levels, palette = "Blues", darker_high = TRUE)
} else {
  # Memory lambda: green gradient, darker for lower lambda.
  numeric_gradient_colors(color_levels, palette = "Greens", darker_high = FALSE)
}
color_cols <- palette_cols
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

color_legend_title <- if (identical(color_by, "opportunity")) "opportunity" else "lambda"
color_legend_labels <- paste(color_legend_title, color_level_labels)
opportunity_pch <- setNames(
  rep(19, length.out = length(opportunity_levels)),
  opportunity_levels
)
opportunity_lty <- setNames(
  rep(1, length.out = length(opportunity_levels)),
  opportunity_levels
)

point_color_for <- function(beta, opportunity, alpha = 0.55) {
  colors <- mapply(series_color, beta, opportunity)
  grDevices::adjustcolor(colors, alpha.f = alpha)
}

line_color_for <- function(beta, opportunity) {
  series_color(beta, opportunity)
}

expand_range <- function(x, pad = 0.05, default = c(0, 1)) {
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

plot_model_legend <- function(x = "topleft", y = NULL, cex = 1) {
  legend_args <- list(
    x = x,
    legend = color_legend_labels,
    col = unname(color_cols[color_levels]),
    pch = rep(19, length(color_legend_labels)),
    lty = rep(1, length(color_legend_labels)),
    bty = "n",
    cex = cex
  )
  if (!is.null(y)) {
    legend_args$y <- y
  }
  do.call(legend, legend_args)
}

draw_legend_panel <- function(include_time_cost = FALSE, cex = 1) {
  par(mar = c(0, 0, 0, 0))
  apply_panel_text_style()
  plot.new()
  plot.window(xlim = c(0, 1), ylim = c(0, 1))
  plot_model_legend(x = 0, y = 0.95, cex = cex)
  if (include_time_cost) {
    plot_time_cost_legend(x = 0, y = 0.58, cex = cex)
  }
}

fmt_num <- function(value) {
  value <- as.numeric(value)
  if (is.na(value)) {
    return("NA")
  }
  if (abs(value - round(value)) < 1e-12) {
    return(as.character(as.integer(round(value))))
  }
  format(value, scientific = FALSE, trim = TRUE)
}

parse_state_label <- function(label) {
  parts <- strsplit(as.character(label), ";", fixed = TRUE)[[1]]
  lapply(parts, function(part) {
    clean <- gsub("^\\[|\\]$", "", part)
    if (!nzchar(clean)) {
      return(numeric())
    }
    suppressWarnings(as.numeric(strsplit(clean, ",", fixed = TRUE)[[1]]))
  })
}

canonical_state_label <- function(path_states) {
  path_states <- lapply(path_states, function(values) sort(as.numeric(values)))
  max_len <- max(vapply(path_states, length, integer(1)), 0L)
  order_df <- data.frame(
    row = seq_along(path_states),
    length = vapply(path_states, length, integer(1))
  )
  if (max_len > 0L) {
    for (value_i in seq_len(max_len)) {
      value <- vapply(path_states, function(values) {
        if (length(values) >= value_i) values[[value_i]] else NA_real_
      }, numeric(1))
      order_df[[paste0("value_", value_i)]] <- value
    }
  }
  order_cols <- c("length", paste0("value_", seq_len(max_len)))
  order_cols <- order_cols[order_cols %in% names(order_df)]
  order_args <- c(order_df[order_cols], list(na.last = TRUE))
  path_states <- path_states[order_df$row[do.call(order, order_args)]]
  paste(
    vapply(path_states, function(values) {
      paste0("[", paste(vapply(values, fmt_num, character(1)), collapse = ","), "]")
    }, character(1)),
    collapse = ";"
  )
}

parse_observe_path_state <- function(value) {
  value <- as.character(value)
  if (!nzchar(value) || is.na(value)) {
    return(NULL)
  }
  clean <- gsub("^\\[|\\]$", "", value)
  if (!nzchar(clean)) {
    return(numeric())
  }
  suppressWarnings(as.numeric(strsplit(clean, ",", fixed = TRUE)[[1]]))
}

path_state_equal <- function(a, b) {
  a <- sort(as.numeric(a))
  b <- sort(as.numeric(b))
  length(a) == length(b) && all(abs(a - b) < 1e-12)
}

exact_file <- function(suffix) {
  file.path(exact_dir, sprintf("exact_time_cost_%s.csv", suffix))
}

zero_exact_file <- function(suffix) {
  file.path(zero_exact_dir, sprintf("exact_time_cost_%s.csv", suffix))
}

requested_time_costs <- suppressWarnings(as.numeric(optimal_time_cost_values))
if (all(!is.na(requested_time_costs))) {
  requested_time_costs <- sort(unique(c(0, requested_time_costs)))
}
keep_plotted_exact_time_cost <- function(time_cost) {
  time_cost_numeric <- suppressWarnings(as.numeric(time_cost))
  is.na(time_cost_numeric) | abs(time_cost_numeric) > 1e-8
}
reward_value_count <- if (identical(input_type, "binary")) 2 else 8
raw_reward_bits_per_observation <- log2(reward_value_count)
raw_reward_nats_per_observation <- log(reward_value_count)

exact_policy_diagnostics_file <- function(suffix) {
  if (!nzchar(task_name)) {
    return(NA_character_)
  }
  file.path(
    exact_dir,
    ".policy_diagnostics",
    sprintf("exact_time_cost_%s_%s.csv", task_name, suffix)
  )
}

cached_exact_diagnostics_available <- function() {
  if (!nzchar(task_name)) {
    return(FALSE)
  }
  file_has_cols <- function(path, cols) {
    if (!file.exists(path)) {
      return(FALSE)
    }
    header <- tryCatch(names(utils::read.csv(path, nrows = 0, check.names = FALSE)), error = function(e) character())
    all(cols %in% header)
  }
  trial_path <- exact_policy_diagnostics_file("trial_summary")
  continue_suffix <- if (task_name %in% c("default2", "disjoint2x2")) {
    "continue_difference_summary"
  } else {
    "continue_best_summary"
  }
  continue_path <- exact_policy_diagnostics_file(continue_suffix)
  continue_cols <- c("time_cost", "decision_timestep", "p_continue", "n")
  if (identical(task_name, "disjoint3x2")) {
    continue_cols <- c(continue_cols, "best_path_value", "best_path_complete")
  } else if (identical(continue_suffix, "continue_best_summary")) {
    continue_cols <- c(continue_cols, "best_path_value")
  } else {
    continue_cols <- c(continue_cols, "path_value_difference")
  }
  file_has_cols(
    trial_path,
    c(
      "time_cost",
      "normalized_chosen_path_reward",
      "observations_before_stop",
      "var_observations_before_stop",
      "cumulative_raw_reward_information_bits"
    )
  ) &&
    file_has_cols(continue_path, continue_cols) &&
    (
      !identical(task_name, "disjoint3x2") ||
        file_has_cols(
          exact_policy_diagnostics_file("condition_action_summary"),
          c(
            "time_cost",
            "decision_timestep",
            "count_pattern",
            "best_path_identity",
            "best_path_value",
            "action_label",
            "p_action",
            "n"
          )
        )
    )
}

use_cached_exact_diagnostics <- cached_exact_diagnostics_available()

exact_breakpoint_task_file <- function(suffix) {
  if (!nzchar(task_name)) {
    return(NA_character_)
  }
  file.path(
    exact_dir,
    ".breakpoint_task_runs",
    sprintf("exact_time_cost_%s_%s.csv", task_name, suffix)
  )
}

available_exact_costs_from_summary <- function(path) {
  if (!file.exists(path) || !nzchar(task_name)) {
    return(numeric())
  }
  dat <- read_csv_fast(path, keep_names = c("task", "time_cost"))
  if (!all(c("task", "time_cost") %in% names(dat))) {
    return(numeric())
  }
  sort(unique(suppressWarnings(as.numeric(dat$time_cost[dat$task == task_name]))))
}

contains_requested_cost <- function(available, requested) {
  if (length(available) == 0 || length(requested) == 0 || any(is.na(requested))) {
    return(rep(FALSE, length(requested)))
  }
  vapply(requested, function(x) any(abs(available - x) < 1e-8), logical(1))
}

read_exact_table <- function(suffix) {
  exact_keep_names <- switch(
    suffix,
    "states" = c(
      "task", "tree_size", "tree_config", "num_paths", "path_length", "n_nodes",
      "time_cost", "reward_values", "reward_prior_mean", "normalize_reward",
      "reward_norm", "min_observations_before_stop", "tie_mode",
      "state_label", "observed_count"
    ),
    "actions" = c(
      "task", "tree_size", "tree_config", "num_paths", "path_length", "n_nodes",
      "time_cost", "reward_values", "reward_prior_mean", "normalize_reward",
      "reward_norm", "min_observations_before_stop", "tie_mode",
      "state_label", "action_kind", "observe_path_state", "q_value"
    ),
    character()
  )
  main_path <- exact_file(suffix)
  zero_path <- zero_exact_file(suffix)
  breakpoint_path <- exact_breakpoint_task_file(suffix)
  paths <- character()
  if (!is.na(breakpoint_path) && file.exists(breakpoint_path)) {
    paths <- c(paths, breakpoint_path)
    breakpoint_summary_path <- exact_breakpoint_task_file("summary")
    main_summary_path <- exact_file("summary")
    breakpoint_costs <- available_exact_costs_from_summary(breakpoint_summary_path)
    main_costs <- available_exact_costs_from_summary(main_summary_path)
    missing_from_breakpoint <- requested_time_costs[
      !contains_requested_cost(breakpoint_costs, requested_time_costs)
    ]
    if (
      file.exists(main_path) &&
        any(contains_requested_cost(main_costs, missing_from_breakpoint))
    ) {
      paths <- c(paths, main_path)
    }
  } else {
    paths <- c(paths, main_path)
  }
  if (file.exists(zero_path)) {
    paths <- c(paths, zero_path)
  }
  existing_paths <- paths[file.exists(paths)]
  if (length(existing_paths) == 0) {
    warning(sprintf(
      "Exact %s file not found at %s or %s; optimal overlay will be skipped.",
      suffix,
      exact_file(suffix),
      ifelse(is.na(breakpoint_path), "<no task fallback>", breakpoint_path)
    ))
    return(NULL)
  }
  tables <- lapply(existing_paths, function(path) {
    dat <- read_csv_fast(path, keep_names = exact_keep_names)
    if (identical(normalizePath(path, mustWork = FALSE), normalizePath(zero_path, mustWork = FALSE))) {
      if (all(c("task", "time_cost") %in% names(dat))) {
        dat <- dat[
          dat$task == task_name &
            abs(suppressWarnings(as.numeric(dat$time_cost))) < 1e-8,
          ,
          drop = FALSE
        ]
      }
    } else if (all(c("time_cost") %in% names(dat))) {
      dat <- dat[
        abs(suppressWarnings(as.numeric(dat$time_cost))) > 1e-8,
        ,
        drop = FALSE
      ]
    }
    dat
  })
  tables <- tables[vapply(tables, nrow, integer(1)) > 0]
  if (length(tables) == 0) {
    return(NULL)
  }
  out <- bind_rows_fill(tables)
  if (length(existing_paths) > 1) {
    out <- unique(out)
  }
  out
}

exact_states <- if (use_cached_exact_diagnostics) NULL else read_exact_table("states")
exact_actions <- if (use_cached_exact_diagnostics) NULL else read_exact_table("actions")
exact_occupancy <- NULL

warn_missing_exact_time_costs <- function(dat) {
  if (is.null(dat) || !nzchar(task_name) || any(is.na(requested_time_costs))) {
    return(invisible(NULL))
  }
  if (!all(c("task", "time_cost") %in% names(dat))) {
    return(invisible(NULL))
  }
  available <- sort(unique(suppressWarnings(as.numeric(dat$time_cost[dat$task == task_name]))))
  if (length(available) == 0) {
    warning(sprintf("No exact rows found for task=%s; optimal overlays will be skipped.", task_name))
    return(invisible(NULL))
  }
  missing <- requested_time_costs[!vapply(requested_time_costs, function(x) {
    any(abs(available - x) < 1e-8)
  }, logical(1))]
  if (length(missing) > 0) {
    warning(sprintf(
      "Exact rows for task=%s are missing requested optimal time costs: %s. Available exact costs: %s. Missing optimal dots/legends will be skipped.",
      task_name,
      paste(format_plot_values(missing), collapse = ", "),
      paste(format_plot_values(available), collapse = ", ")
    ))
  }
  invisible(NULL)
}

warn_missing_exact_time_costs(exact_actions)

filter_exact <- function(dat) {
  if (is.null(dat) || !nzchar(task_name)) {
    return(NULL)
  }
  if (!"task" %in% names(dat) || !"time_cost" %in% names(dat)) {
    return(NULL)
  }
  keep <- dat$task == task_name
  if (all(!is.na(requested_time_costs))) {
    keep <- keep & vapply(dat$time_cost, function(x) {
      any(abs(as.numeric(x) - requested_time_costs) < 1e-8)
    }, logical(1))
  }
  keep <- keep & keep_plotted_exact_time_cost(dat$time_cost)
  out <- dat[keep, , drop = FALSE]
  if (nrow(out) == 0) {
    return(NULL)
  }
  out
}

exact_states_task <- filter_exact(exact_states)
exact_actions_task <- filter_exact(exact_actions)
exact_occupancy_task <- filter_exact(exact_occupancy)

exact_action_lookup_key <- function(state_label, time_cost) {
  paste(as.character(state_label), format_plot_values(time_cost), sep = "\r")
}

exact_actions_by_state_cost <- if (!is.null(exact_actions_task) && nrow(exact_actions_task) > 0) {
  split(
    exact_actions_task,
    exact_action_lookup_key(exact_actions_task$state_label, exact_actions_task$time_cost)
  )
} else {
  list()
}

cached_exact_time_cost_values <- function() {
  path <- exact_policy_diagnostics_file("trial_summary")
  if (is.na(path) || !file.exists(path)) {
    return(numeric())
  }
  dat <- read_csv_fast(path, keep_names = c("time_cost"))
  if (!"time_cost" %in% names(dat)) {
    return(numeric())
  }
  values <- sort(unique(suppressWarnings(as.numeric(dat$time_cost))))
  if (all(!is.na(requested_time_costs))) {
    values <- values[vapply(values, function(x) {
      any(abs(x - requested_time_costs) < 1e-8)
    }, logical(1))]
  }
  values <- values[keep_plotted_exact_time_cost(values)]
  values
}

exact_time_cost_values <- if (!is.null(exact_actions_task) && nrow(exact_actions_task) > 0) {
  sort(unique(suppressWarnings(as.numeric(exact_actions_task$time_cost))))
} else {
  cached_exact_time_cost_values()
}
exact_time_cost_values <- exact_time_cost_values[keep_plotted_exact_time_cost(exact_time_cost_values)]
exact_time_cost_levels <- format_plot_values(exact_time_cost_values)
exact_time_cost_cols <- numeric_gradient_colors(
  exact_time_cost_levels,
  palette = "Purples",
  darker_high = TRUE
)

time_cost_key <- function(value) {
  format_plot_values(value)
}

time_cost_color_for <- function(time_cost, alpha = 1) {
  keys <- time_cost_key(time_cost)
  cols <- unname(exact_time_cost_cols[keys])
  cols[is.na(cols)] <- "black"
  grDevices::adjustcolor(cols, alpha.f = alpha)
}

exact_policy_diagnostics_file <- function(suffix) {
  if (!nzchar(task_name)) {
    return(NA_character_)
  }
  file.path(
    exact_dir,
    ".policy_diagnostics",
    sprintf("exact_time_cost_%s_%s.csv", task_name, suffix)
  )
}

filter_requested_exact_costs <- function(dat) {
  if (is.null(dat) || nrow(dat) == 0 || !"time_cost" %in% names(dat)) {
    return(dat)
  }
  keep <- keep_plotted_exact_time_cost(dat$time_cost)
  if (all(!is.na(requested_time_costs))) {
    keep <- keep & vapply(dat$time_cost, function(x) {
      any(abs(suppressWarnings(as.numeric(x)) - requested_time_costs) < 1e-8)
    }, logical(1))
  }
  dat[keep, , drop = FALSE]
}

read_exact_policy_diagnostics <- function(suffix, required_cols) {
  path <- exact_policy_diagnostics_file(suffix)
  if (is.na(path) || !file.exists(path)) {
    return(NULL)
  }
  dat <- read_csv_fast(path, keep_names = required_cols)
  if (!all(required_cols %in% names(dat))) {
    warning(sprintf(
      "Cached exact diagnostic %s is missing required columns; falling back to R computation.",
      path
    ))
    return(NULL)
  }
  dat$time_cost <- suppressWarnings(as.numeric(dat$time_cost))
  dat <- filter_requested_exact_costs(dat)
  if (nrow(dat) == 0) {
    return(NULL)
  }
  message(sprintf("Using cached exact diagnostic summary from %s", path))
  dat
}

plot_time_cost_legend <- function(x = "topleft", y = NULL, cex = 1) {
  if (length(exact_time_cost_levels) == 0) {
    return(invisible(NULL))
  }
  legend_args <- list(
    x = x,
    legend = paste("time cost", exact_time_cost_levels),
    col = unname(exact_time_cost_cols[exact_time_cost_levels]),
    pch = rep(17, length(exact_time_cost_levels)),
    lty = rep(1, length(exact_time_cost_levels)),
    bty = "n",
    cex = cex
  )
  if (!is.null(y)) {
    legend_args$y <- y
  }
  do.call(legend, legend_args)
}

exact_p_continue_stop_prefer_from_actions <- function(state_actions, tol = 1e-10) {
  if (is.null(state_actions) || nrow(state_actions) == 0) {
    return(NA_real_)
  }
  q_values <- suppressWarnings(as.numeric(state_actions$q_value))
  if (all(is.na(q_values))) {
    return(NA_real_)
  }
  best_q <- max(q_values, na.rm = TRUE)
  is_best <- abs(q_values - best_q) <= tol
  if (any(as.character(state_actions$action_kind) == "stop" & is_best, na.rm = TRUE)) {
    return(0)
  }
  if (any(as.character(state_actions$action_kind) == "observe" & is_best, na.rm = TRUE)) {
    return(1)
  }
  NA_real_
}

exact_stop_prefer_action <- function(state_actions, tol = 1e-10) {
  if (is.null(state_actions) || nrow(state_actions) == 0) {
    return(NULL)
  }
  q_values <- suppressWarnings(as.numeric(state_actions$q_value))
  if (all(is.na(q_values))) {
    return(NULL)
  }
  best_q <- max(q_values, na.rm = TRUE)
  is_best <- abs(q_values - best_q) <= tol
  stop_rows <- state_actions[as.character(state_actions$action_kind) == "stop" & is_best, , drop = FALSE]
  if (nrow(stop_rows) > 0) {
    return(stop_rows[1, , drop = FALSE])
  }
  observe_rows <- state_actions[as.character(state_actions$action_kind) == "observe" & is_best, , drop = FALSE]
  if (nrow(observe_rows) > 0) {
    return(observe_rows[1, , drop = FALSE])
  }
  NULL
}

exact_p_continue_stop_prefer <- function(state_label, time_cost) {
  if (length(exact_actions_by_state_cost) == 0) {
    return(NA_real_)
  }
  state_actions <- exact_actions_by_state_cost[[exact_action_lookup_key(state_label, time_cost)]]
  exact_p_continue_stop_prefer_from_actions(state_actions)
}

simulate_exact_trials_for_cost <- function(time_cost) {
  if (is.null(exact_actions_task) || !nzchar(task_name)) {
    return(data.frame())
  }
  actions <- exact_actions_task[
    abs(as.numeric(exact_actions_task$time_cost) - as.numeric(time_cost)) < 1e-8,
    ,
    drop = FALSE
  ]
  if (nrow(actions) == 0) {
    return(data.frame())
  }
  reward_values <- suppressWarnings(as.numeric(strsplit(actions$reward_values[[1]], ",", fixed = TRUE)[[1]]))
  if (any(is.na(reward_values))) {
    return(data.frame())
  }
  actions_by_state <- split(actions, actions$state_label)

  assignments <- expand.grid(
    rep(list(reward_values), task_node_count),
    KEEP.OUT.ATTRS = FALSE,
    stringsAsFactors = FALSE
  )
  rows <- vector("list", nrow(assignments))
  path_nodes <- split(seq_len(task_node_count), rep(seq_len(task_path_count), each = task_nodes_per_path))

  for (assignment_i in seq_len(nrow(assignments))) {
    rewards <- as.numeric(assignments[assignment_i, ])
    observed_path_values <- vector("list", task_path_count)
    observed_path_node_counts <- rep(0L, task_path_count)
    observed_nodes_by_path <- vector("list", task_path_count)
    for (path_i in seq_len(task_path_count)) {
      observed_path_values[[path_i]] <- numeric()
      observed_nodes_by_path[[path_i]] <- integer()
    }
    observations <- 0L

    repeat {
      state_label <- canonical_state_label(observed_path_values)
      state_actions <- actions_by_state[[state_label]]
      if (is.null(state_actions) || nrow(state_actions) == 0) {
        break
      }
      action <- exact_stop_prefer_action(state_actions)
      if (is.null(action)) {
        break
      }
      if (identical(as.character(action$action_kind[[1]]), "stop") || observations >= task_node_count) {
        break
      }
      target_path_state <- parse_observe_path_state(action$observe_path_state[[1]])
      selected_path <- NA_integer_
      for (path_i in seq_len(task_path_count)) {
        if (
          observed_path_node_counts[[path_i]] < task_nodes_per_path &&
            path_state_equal(observed_path_values[[path_i]], target_path_state)
        ) {
          selected_path <- path_i
          break
        }
      }
      if (is.na(selected_path)) {
        break
      }
      candidate_nodes <- setdiff(path_nodes[[selected_path]], observed_nodes_by_path[[selected_path]])
      selected_node <- candidate_nodes[[1]]
      observed_nodes_by_path[[selected_path]] <- c(observed_nodes_by_path[[selected_path]], selected_node)
      observed_path_values[[selected_path]] <- c(observed_path_values[[selected_path]], rewards[[selected_node]])
      observed_path_node_counts[[selected_path]] <- observed_path_node_counts[[selected_path]] + 1L
      observations <- observations + 1L
    }

    path_rewards <- vapply(path_nodes, function(nodes) sum(rewards[nodes]), numeric(1))
    prior_mean <- mean(reward_values)
    reward_bits <- log2(length(reward_values))
    reward_nats <- log(length(reward_values))
    cumulative_observation_count <- observations * (observations + 1) / 2
    path_posterior_values <- vapply(seq_len(task_path_count), function(path_i) {
      sum(observed_path_values[[path_i]]) +
        (task_nodes_per_path - observed_path_node_counts[[path_i]]) * prior_mean
    }, numeric(1))
    chosen_path <- which(abs(path_posterior_values - max(path_posterior_values)) < 1e-12)[[1]]
    chosen_path_reward <- path_rewards[[chosen_path]]
    rows[[assignment_i]] <- data.frame(
      time_cost = time_cost,
      opportunity = as.character(time_cost),
      chosen_path_reward = chosen_path_reward,
      normalized_chosen_path_reward = normalize_chosen_path_reward(chosen_path_reward, path_rewards),
      observations_before_stop = observations,
      final_raw_reward_information_bits = observations * reward_bits,
      final_raw_reward_information_nats = observations * reward_nats,
      cumulative_raw_reward_information_bits = cumulative_observation_count * reward_bits,
      cumulative_raw_reward_information_nats = cumulative_observation_count * reward_nats,
      kl_paid_total = 0,
      stringsAsFactors = FALSE
    )
  }
  do.call(rbind, rows)
}

build_exact_trial_overlay <- function() {
  cached <- read_exact_policy_diagnostics(
    "trial_summary",
    c(
      "time_cost", "kl_paid_total", "chosen_path_reward",
      "normalized_chosen_path_reward", "observations_before_stop",
      "var_observations_before_stop",
      "final_raw_reward_information_bits",
      "final_raw_reward_information_nats",
      "cumulative_raw_reward_information_bits",
      "cumulative_raw_reward_information_nats",
      "n"
    )
  )
  if (!is.null(cached)) {
    if (any(as.character(cached$action_label) == "stop", na.rm = TRUE)) {
      message("Cached disjoint3x2 exact condition/action summary has unlabeled stop rows; rebuilding from exact actions.")
    } else {
      return(cached)
    }
  }
  if (is.null(exact_actions_task) || any(is.na(requested_time_costs))) {
    return(data.frame())
  }
  available_time_costs <- sort(unique(suppressWarnings(as.numeric(exact_actions_task$time_cost))))
  rows <- lapply(available_time_costs, simulate_exact_trials_for_cost)
  rows <- rows[vapply(rows, nrow, integer(1)) > 0]
  if (length(rows) == 0) {
    return(data.frame())
  }
  do.call(rbind, rows)
}

build_exact_continue_summary <- function(feature_type) {
  cached_suffix <- if (identical(feature_type, "difference")) {
    "continue_difference_summary"
  } else {
    "continue_best_summary"
  }
  cached_feature_col <- if (identical(feature_type, "difference")) {
    "path_value_difference"
  } else {
    "best_path_value"
  }
  cached_required_cols <- c("time_cost", "decision_timestep", cached_feature_col, "p_continue", "n")
  if (identical(feature_type, "best") && is_disjoint3x2) {
    cached_required_cols <- c(cached_required_cols, "best_path_complete")
  }
  cached <- read_exact_policy_diagnostics(cached_suffix, cached_required_cols)
  if (!is.null(cached)) {
    return(cached)
  }
  if (identical(feature_type, "difference")) {
    return(build_exact_signed_difference_continue_summary())
  }
  if (is.null(exact_states_task)) {
    return(data.frame())
  }
  rows <- list()
  for (row_i in seq_len(nrow(exact_states_task))) {
    observed_count_i <- suppressWarnings(as.integer(exact_states_task$observed_count[[row_i]]))
    if (is.na(observed_count_i) || observed_count_i <= 0L || observed_count_i >= task_node_count) {
      next
    }
    state <- parse_state_label(exact_states_task$state_label[[row_i]])
    path_values <- vapply(state, sum, numeric(1))
    path_counts <- vapply(state, length, integer(1))
    if (feature_type == "best") {
      observed_values <- path_values[path_counts > 0L]
      if (length(observed_values) == 0) {
        next
      }
      feature_value <- max(observed_values)
      feature_col <- "best_path_value"
      best_path_complete <- if (is_disjoint3x2) {
        best_indices <- which(path_counts > 0L & abs(path_values - feature_value) < 1e-12)
        if (any(path_counts[best_indices] >= task_nodes_per_path)) "complete" else "incomplete"
      } else {
        NA_character_
      }
    } else {
      if (length(path_values) != 2L) {
        next
      }
      feature_value <- max(path_values) - min(path_values)
      feature_col <- "path_value_difference"
      best_path_complete <- NA_character_
    }
    rows[[length(rows) + 1]] <- data.frame(
      time_cost = exact_states_task$time_cost[[row_i]],
      decision_timestep = observed_count_i + 1L,
      feature_value = feature_value,
      best_path_complete = best_path_complete,
      state_mass = 1.0,
      p_continue = exact_p_continue_stop_prefer(
        exact_states_task$state_label[[row_i]],
        exact_states_task$time_cost[[row_i]]
      ),
      stringsAsFactors = FALSE
    )
    names(rows[[length(rows)]])[names(rows[[length(rows)]]) == "feature_value"] <- feature_col
  }
  if (length(rows) == 0) {
    return(data.frame())
  }
  dat <- do.call(rbind, rows)
  feature_col <- if (feature_type == "best") "best_path_value" else "path_value_difference"
  group_cols <- c("time_cost", "decision_timestep", feature_col)
  if (feature_type == "best" && is_disjoint3x2 && "best_path_complete" %in% names(dat)) {
    dat <- dat[!is.na(dat$best_path_complete), , drop = FALSE]
    group_cols <- c(group_cols, "best_path_complete")
  }
  split_keys <- do.call(interaction, c(dat[, group_cols, drop = FALSE], list(drop = TRUE)))
  summary_rows <- lapply(split(dat, split_keys), function(piece) {
    denom <- sum(piece$state_mass, na.rm = TRUE)
    if (denom <= 0) {
      return(NULL)
    }
    out <- data.frame(
      time_cost = piece$time_cost[[1]],
      decision_timestep = piece$decision_timestep[[1]],
      p_continue = sum(piece$p_continue * piece$state_mass, na.rm = TRUE) / denom,
      n = denom,
      stringsAsFactors = FALSE
    )
    out[[feature_col]] <- piece[[feature_col]][[1]]
    if ("best_path_complete" %in% group_cols) {
      out$best_path_complete <- piece$best_path_complete[[1]]
    }
    out[, c(group_cols, "p_continue", "n"), drop = FALSE]
  })
  summary_rows <- summary_rows[!vapply(summary_rows, is.null, logical(1))]
  do.call(rbind, summary_rows)
}

disjoint3x2_exact_action_label <- function(action, path_states, path_order) {
  action_kind <- as.character(action$action_kind[[1]])
  if (identical(action_kind, "stop")) {
    prior_mean <- mean(reward_values)
    posterior_values <- vapply(path_states, function(path_state) {
      sum(as.numeric(path_state)) + (task_nodes_per_path - length(path_state)) * prior_mean
    }, numeric(1))
    ordered_values <- posterior_values[path_order]
    best_rank <- which(abs(ordered_values - max(ordered_values, na.rm = TRUE)) < 1e-12)[[1]]
    return(paste0("stop_P", best_rank))
  }
  if (!identical(action_kind, "observe")) {
    return(if (nzchar(action_kind)) action_kind else "unknown")
  }
  target_path_state <- parse_observe_path_state(action$observe_path_state[[1]])
  for (rank_i in seq_along(path_order)) {
    path_i <- path_order[[rank_i]]
    if (
      length(path_states[[path_i]]) < task_nodes_per_path &&
        path_state_equal(path_states[[path_i]], target_path_state)
    ) {
      return(paste0("observe_P", rank_i))
    }
  }
  "observe_unknown"
}

build_exact_disjoint3x2_condition_action_summary <- function() {
  if (!is_disjoint3x2) {
    return(data.frame())
  }
  cached <- read_exact_policy_diagnostics(
    "condition_action_summary",
    c(
      "time_cost",
      "decision_timestep",
      "count_pattern",
      "best_path_identity",
      "best_path_value",
      "action_label",
      "p_action",
      "n"
    )
  )
  if (!is.null(cached)) {
    return(cached)
  }
  if (is.null(exact_states_task) || length(exact_actions_by_state_cost) == 0) {
    return(data.frame())
  }
  rows <- list()
  for (row_i in seq_len(nrow(exact_states_task))) {
    observed_count_i <- suppressWarnings(as.integer(exact_states_task$observed_count[[row_i]]))
    if (is.na(observed_count_i) || observed_count_i <= 0L || observed_count_i >= task_node_count) {
      next
    }
    path_states <- parse_state_label(exact_states_task$state_label[[row_i]])
    if (length(path_states) != task_path_count) {
      next
    }
    path_values <- vapply(path_states, sum, numeric(1))
    path_counts <- vapply(path_states, length, integer(1))
    condition <- disjoint3x2_condition_from_state(path_values, path_counts)
    state_actions <- exact_actions_by_state_cost[[
      exact_action_lookup_key(
        exact_states_task$state_label[[row_i]],
        exact_states_task$time_cost[[row_i]]
      )
    ]]
    action <- exact_stop_prefer_action(state_actions)
    if (is.null(action)) {
      next
    }
    rows[[length(rows) + 1L]] <- data.frame(
      time_cost = exact_states_task$time_cost[[row_i]],
      decision_timestep = observed_count_i + 1L,
      count_pattern = condition$count_pattern,
      best_path_identity = condition$best_path_identity,
      best_path_value = condition$best_path_value,
      action_label = disjoint3x2_exact_action_label(action, path_states, condition$path_order),
      mass = 1.0,
      stringsAsFactors = FALSE
    )
  }
  if (length(rows) == 0) {
    return(data.frame())
  }
  action_data <- do.call(rbind, rows)
  group_cols <- c(
    "time_cost",
    "decision_timestep",
    "count_pattern",
    "best_path_identity",
    "best_path_value"
  )
  action_counts <- aggregate(
    mass ~ time_cost + decision_timestep + count_pattern + best_path_identity + best_path_value + action_label,
    data = action_data,
    FUN = sum
  )
  names(action_counts)[names(action_counts) == "mass"] <- "n"
  total_counts <- aggregate(
    mass ~ time_cost + decision_timestep + count_pattern + best_path_identity + best_path_value,
    data = action_data,
    FUN = sum
  )
  names(total_counts)[names(total_counts) == "mass"] <- "total_n"
  out <- merge(action_counts, total_counts, by = group_cols)
  out$p_action <- out$n / out$total_n
  out[, c(group_cols, "action_label", "p_action", "n"), drop = FALSE]
}

build_exact_signed_difference_continue_summary_for_cost <- function(time_cost) {
  if (is.null(exact_actions_task) || task_path_count != 2L) {
    return(data.frame())
  }
  actions <- exact_actions_task[
    abs(as.numeric(exact_actions_task$time_cost) - as.numeric(time_cost)) < 1e-8,
    ,
    drop = FALSE
  ]
  if (nrow(actions) == 0) {
    return(data.frame())
  }
  reward_values <- suppressWarnings(as.numeric(strsplit(actions$reward_values[[1]], ",", fixed = TRUE)[[1]]))
  if (any(is.na(reward_values))) {
    return(data.frame())
  }
  actions_by_state <- split(actions, actions$state_label)

  ordered_node_sequences <- function(nodes, k) {
    if (k == 0L) {
      return(list(integer()))
    }
    out <- list()
    for (node in nodes) {
      suffixes <- ordered_node_sequences(setdiff(nodes, node), k - 1L)
      for (suffix in suffixes) {
        out[[length(out) + 1]] <- c(node, suffix)
      }
    }
    out
  }

  rows <- list()
  for (decision_timestep in 2:task_node_count) {
    observed_count <- decision_timestep - 1L
    node_sequences <- ordered_node_sequences(seq_len(task_node_count), observed_count)
    reward_assignments <- expand.grid(
      rep(list(reward_values), observed_count),
      KEEP.OUT.ATTRS = FALSE,
      stringsAsFactors = FALSE
    )

    for (node_sequence in node_sequences) {
      for (assignment_i in seq_len(nrow(reward_assignments))) {
        observed_rewards <- as.numeric(reward_assignments[assignment_i, ])
        observed_values <- vector("list", task_path_count)
        for (path_i in seq_len(task_path_count)) {
          observed_values[[path_i]] <- numeric()
        }
        for (obs_i in seq_along(node_sequence)) {
          path_i <- path_id_for_node(node_sequence[[obs_i]])
          observed_values[[path_i]] <- c(observed_values[[path_i]], observed_rewards[[obs_i]])
        }

        state_label <- canonical_state_label(observed_values)
        state_actions <- actions_by_state[[state_label]]
        if (is.null(state_actions) || nrow(state_actions) == 0) {
          next
        }
        p_continue <- exact_p_continue_stop_prefer_from_actions(state_actions)
        if (is.na(p_continue)) {
          next
        }

        current_path <- path_id_for_node(node_sequence[[length(node_sequence)]])
        other_path <- if (current_path == 1L) 2L else 1L
        rows[[length(rows) + 1]] <- data.frame(
          time_cost = time_cost,
          decision_timestep = decision_timestep,
          path_value_difference = sum(observed_values[[current_path]]) -
            sum(observed_values[[other_path]]),
          p_continue = p_continue,
          mass = 1.0,
          stringsAsFactors = FALSE
        )
      }
    }
  }

  if (length(rows) == 0) {
    return(data.frame())
  }
  decision_data <- do.call(rbind, rows)
  split_keys <- interaction(
    decision_data$time_cost,
    decision_data$decision_timestep,
    decision_data$path_value_difference,
    drop = TRUE
  )
  summary_rows <- lapply(split(decision_data, split_keys), function(piece) {
    denom <- sum(piece$mass, na.rm = TRUE)
    if (denom <= 0) {
      return(NULL)
    }
    data.frame(
      time_cost = piece$time_cost[[1]],
      decision_timestep = piece$decision_timestep[[1]],
      path_value_difference = piece$path_value_difference[[1]],
      p_continue = sum(piece$p_continue * piece$mass, na.rm = TRUE) / denom,
      n = denom,
      stringsAsFactors = FALSE
    )
  })
  summary_rows <- summary_rows[!vapply(summary_rows, is.null, logical(1))]
  do.call(rbind, summary_rows)
}

build_exact_signed_difference_continue_summary <- function() {
  if (is.null(exact_actions_task) || any(is.na(requested_time_costs))) {
    return(data.frame())
  }
  available_time_costs <- sort(unique(suppressWarnings(as.numeric(exact_actions_task$time_cost))))
  rows <- lapply(available_time_costs, build_exact_signed_difference_continue_summary_for_cost)
  rows <- rows[vapply(rows, nrow, integer(1)) > 0]
  if (length(rows) == 0) {
    return(data.frame())
  }
  do.call(rbind, rows)
}

trial_diagnostics <- build_trial_diagnostics(all_data)
exact_trial_overlay <- build_exact_trial_overlay()

population_var <- function(x) {
  x <- as.numeric(x)
  x <- x[is.finite(x)]
  if (length(x) == 0) {
    return(NA_real_)
  }
  mean((x - mean(x))^2)
}

mean_or_na <- function(x) {
  x <- as.numeric(x)
  x <- x[is.finite(x)]
  if (length(x) == 0) {
    return(NA_real_)
  }
  mean(x)
}

aggregate_means_by <- function(dat, group_cols, value_cols) {
  pieces <- lapply(value_cols, function(value_col) {
    aggregate(
      dat[[value_col]],
      by = dat[, group_cols, drop = FALSE],
      FUN = mean_or_na
    )
  })
  for (i in seq_along(pieces)) {
    names(pieces[[i]])[names(pieces[[i]]) == "x"] <- value_cols[[i]]
  }
  Reduce(function(left, right) merge(left, right, by = group_cols, all = TRUE), pieces)
}

add_raw_information_columns <- function(dat) {
  if (nrow(dat) == 0 || !"observations_before_stop" %in% names(dat)) {
    return(dat)
  }
  observations <- as.numeric(dat$observations_before_stop)
  observations[!is.finite(observations)] <- 0
  dat$observations_before_stop <- observations
  cumulative_observation_count <- observations * (observations + 1) / 2
  if (!"final_raw_reward_information_bits" %in% names(dat)) {
    dat$final_raw_reward_information_bits <- observations * raw_reward_bits_per_observation
  }
  if (!"final_raw_reward_information_nats" %in% names(dat)) {
    dat$final_raw_reward_information_nats <- observations * raw_reward_nats_per_observation
  }
  if (!"cumulative_raw_reward_information_bits" %in% names(dat)) {
    dat$cumulative_raw_reward_information_bits <- cumulative_observation_count * raw_reward_bits_per_observation
  }
  if (!"cumulative_raw_reward_information_nats" %in% names(dat)) {
    dat$cumulative_raw_reward_information_nats <- cumulative_observation_count * raw_reward_nats_per_observation
  }
  dat
}

build_average_kl_reward_summary <- function(dat) {
  if (nrow(dat) == 0) {
    return(data.frame())
  }
  dat <- add_raw_information_columns(dat)
  if ("chosen_path_reward" %in% names(dat)) {
    dat$normalized_chosen_path_reward <- dat$chosen_path_reward / task_reward_norm
  }
  summary_data <- aggregate_means_by(
    dat,
    group_cols = c("sigma", "beta", "opportunity"),
    value_cols = c(
      "kl_paid_total",
      "chosen_path_reward",
      "normalized_chosen_path_reward",
      "observations_before_stop",
      "final_raw_reward_information_bits",
      "final_raw_reward_information_nats",
      "cumulative_raw_reward_information_bits",
      "cumulative_raw_reward_information_nats"
    )
  )
  var_data <- aggregate(
    observations_before_stop ~ sigma + beta + opportunity,
    data = dat,
    FUN = population_var
  )
  names(var_data)[names(var_data) == "observations_before_stop"] <- "var_observations_before_stop"
  count_data <- aggregate(
    kl_paid_total ~ sigma + beta + opportunity,
    data = dat,
    FUN = length
  )
  names(count_data)[names(count_data) == "kl_paid_total"] <- "n"
  merge(
    merge(summary_data, var_data, by = c("sigma", "beta", "opportunity")),
    count_data,
    by = c("sigma", "beta", "opportunity")
  )
}

build_exact_average_kl_reward_summary <- function(dat) {
  if (nrow(dat) == 0 || !"normalized_chosen_path_reward" %in% names(dat) || !"time_cost" %in% names(dat)) {
    return(data.frame())
  }
  dat <- add_raw_information_columns(dat)
  if ("chosen_path_reward" %in% names(dat)) {
    dat$normalized_chosen_path_reward <- dat$chosen_path_reward / task_reward_norm
  }
  if (all(c(
    "var_observations_before_stop",
    "cumulative_raw_reward_information_bits",
    "cumulative_raw_reward_information_nats",
    "n"
  ) %in% names(dat))) {
    dat <- add_raw_information_columns(dat)
    return(dat)
  }
  summary_data <- aggregate_means_by(
    dat,
    group_cols = "time_cost",
    value_cols = c(
      "kl_paid_total",
      "chosen_path_reward",
      "normalized_chosen_path_reward",
      "observations_before_stop",
      "final_raw_reward_information_bits",
      "final_raw_reward_information_nats",
      "cumulative_raw_reward_information_bits",
      "cumulative_raw_reward_information_nats"
    )
  )
  var_data <- aggregate(
    observations_before_stop ~ time_cost,
    data = dat,
    FUN = population_var
  )
  names(var_data)[names(var_data) == "observations_before_stop"] <- "var_observations_before_stop"
  count_data <- aggregate(
    kl_paid_total ~ time_cost,
    data = dat,
    FUN = length
  )
  names(count_data)[names(count_data) == "kl_paid_total"] <- "n"
  merge(
    merge(summary_data, var_data, by = "time_cost"),
    count_data,
    by = "time_cost"
  )
}

jitter_duplicate_points <- function(x, y, x_amount, y_amount) {
  out <- list(x = as.numeric(x), y = as.numeric(y))
  if (length(out$x) <= 1) {
    return(out)
  }
  keys <- paste(signif(out$x, 10), signif(out$y, 10), sep = "\r")
  for (key in unique(keys)) {
    indices <- which(keys == key)
    n_indices <- length(indices)
    if (n_indices <= 1) {
      next
    }
    angles <- seq(0, 2 * pi, length.out = n_indices + 1L)[seq_len(n_indices)]
    radius_scale <- seq(0.65, 1.0, length.out = n_indices)
    out$x[indices] <- out$x[indices] + cos(angles) * x_amount * radius_scale
    out$y[indices] <- out$y[indices] + sin(angles) * y_amount * radius_scale
  }
  out
}

plot_average_summary_scatter <- function(
  summary_data,
  exact_summary,
  x_col,
  y_col,
  xlab,
  ylab,
  file_prefix,
  include_exact = FALSE,
  jitter_duplicates = FALSE
) {
  pdf_path <- file.path(
    results_dir,
    sprintf(
      "%s_%s_loss_scale_%s_alpha_%s_lambda_%s_opportunity_%s_expansion_%s_%s%s%s.png",
      file_prefix,
      input_type, lambda_label, alpha_label, beta_label, opportunity_label,
      expansion_label, tree_file_label, sigma_file_suffix, simulation_source_suffix
    )
  )
  if (include_exact && nzchar(optimal_time_cost_suffix)) {
    pdf_path <- sub("\\.png$", paste0(optimal_time_cost_suffix, ".png"), pdf_path)
  }
  if (include_exact && nrow(exact_summary) > 0 && "time_cost" %in% names(exact_summary)) {
    exact_time_cost_numeric <- suppressWarnings(as.numeric(exact_summary$time_cost))
    exact_summary <- exact_summary[
      is.na(exact_time_cost_numeric) | abs(exact_time_cost_numeric) > 1e-8,
      ,
      drop = FALSE
    ]
  }
  n_sigma <- max(1L, length(sigma_levels))
  open_panel_png(pdf_path, n_cols = n_sigma, n_rows = 1L)
  old_par <- par(no.readonly = TRUE)
  par(mfrow = c(1L, n_sigma))
  par(mar = c(4.2, 4.2, if (n_sigma > 1L) 2.0 else 1, 1))
  apply_panel_text_style()
  has_exact_points <- include_exact &&
    nrow(exact_summary) > 0 &&
    all(c(x_col, y_col, "time_cost") %in% names(exact_summary))
  exact_x <- if (has_exact_points) as.numeric(exact_summary[[x_col]]) else numeric()
  exact_y <- if (has_exact_points) as.numeric(exact_summary[[y_col]]) else numeric()
  global_x_values <- c(as.numeric(summary_data[[x_col]]), exact_x)
  global_y_values <- c(as.numeric(summary_data[[y_col]]), exact_y)
  x_limits <- expand_range(global_x_values, pad = 0.05)
  y_limits <- expand_range(global_y_values, pad = 0.05)
  for (sigma_value in sigma_levels) {
    summary_piece <- filter_sigma_rows(summary_data, sigma_value)
    model_x <- as.numeric(summary_piece[[x_col]])
    model_y <- as.numeric(summary_piece[[y_col]])
    panel_exact_x <- exact_x
    panel_exact_y <- exact_y
    if (jitter_duplicates) {
      x_values <- c(model_x, panel_exact_x)
      y_values <- c(model_y, panel_exact_y)
      x_span <- diff(expand_range(global_x_values, pad = 0.05))
      y_span <- diff(expand_range(global_y_values, pad = 0.05))
      jittered <- jitter_duplicate_points(
        x_values,
        y_values,
        x_amount = 0.012 * x_span,
        y_amount = 0.018 * y_span
      )
      model_x <- jittered$x[seq_along(model_x)]
      model_y <- jittered$y[seq_along(model_y)]
      if (length(panel_exact_x) > 0) {
        exact_indices <- seq_along(panel_exact_x) + nrow(summary_piece)
        panel_exact_x <- jittered$x[exact_indices]
        panel_exact_y <- jittered$y[exact_indices]
      }
    }
    plot(
      NA,
      xlim = x_limits,
      ylim = y_limits,
      xlab = xlab,
      ylab = ylab,
      main = if (n_sigma > 1L) sigma_panel_title(sigma_value) else ""
    )
    grid()
    if (has_exact_points) {
      points(
        panel_exact_x,
        panel_exact_y,
        pch = 17,
        cex = 1.35,
        col = time_cost_color_for(exact_summary$time_cost, alpha = 0.45)
      )
    }
    points(
      model_x,
      model_y,
      pch = opportunity_pch[as.character(summary_piece$opportunity)],
      cex = 1.35,
      col = point_color_for(summary_piece$beta, summary_piece$opportunity, alpha = 0.45)
    )
  }
  par(old_par)
  dev.off()
  message(sprintf("Saved %s", pdf_path))
}

average_kl_reward_summary <- build_average_kl_reward_summary(trial_diagnostics)
exact_average_kl_reward_summary <- build_exact_average_kl_reward_summary(exact_trial_overlay)
# Superseded by average_raw_information_paid_vs_average_normalized_chosen_path_reward.
plot_average_summary_scatter(
  average_kl_reward_summary,
  exact_average_kl_reward_summary,
  x_col = "normalized_chosen_path_reward",
  y_col = "observations_before_stop",
  xlab = "Average normalized chosen path reward",
  ylab = "Average timestep before stopping",
  file_prefix = "average_timestep_before_stop_vs_average_normalized_chosen_path_reward",
  include_exact = TRUE,
  jitter_duplicates = TRUE
)
plot_average_summary_scatter(
  average_kl_reward_summary,
  exact_average_kl_reward_summary,
  x_col = "observations_before_stop",
  y_col = "kl_paid_total",
  xlab = "Average timestep before stopping",
  ylab = "Average KL paid across all timesteps",
  file_prefix = "average_kl_paid_vs_average_timestep_before_stop",
  include_exact = FALSE
)
plot_average_summary_scatter(
  average_kl_reward_summary,
  exact_average_kl_reward_summary,
  x_col = "observations_before_stop",
  y_col = "cumulative_raw_reward_information_bits",
  xlab = "Average timestep before stopping",
  ylab = "Cumulative raw reward information (bits)",
  file_prefix = "average_raw_information_paid_vs_average_timestep_before_stop",
  include_exact = TRUE,
  jitter_duplicates = TRUE
)
plot_average_summary_scatter(
  average_kl_reward_summary,
  exact_average_kl_reward_summary,
  x_col = "normalized_chosen_path_reward",
  y_col = "cumulative_raw_reward_information_bits",
  xlab = "Average normalized chosen path reward",
  ylab = "Cumulative raw reward information (bits)",
  file_prefix = "average_raw_information_paid_vs_average_normalized_chosen_path_reward",
  include_exact = TRUE,
  jitter_duplicates = TRUE
)

plot_continue_summary <- function(summary_data, exact_summary, feature_col, xlab, file_prefix) {
  if (nrow(summary_data) == 0) {
    warning(sprintf("No model rows for %s; skipping plot.", file_prefix))
    return(invisible(NULL))
  }
  condition_col <- if ("best_path_complete" %in% names(summary_data)) "best_path_complete" else NULL
  if (!is.null(condition_col)) {
    summary_data <- summary_data[!is.na(summary_data[[condition_col]]), , drop = FALSE]
    if (!is.null(exact_summary) && nrow(exact_summary) > 0 && condition_col %in% names(exact_summary)) {
      exact_summary <- exact_summary[!is.na(exact_summary[[condition_col]]), , drop = FALSE]
    }
  }
  pdf_path <- file.path(
    results_dir,
    sprintf(
      "%s_%s_loss_scale_%s_alpha_%s_lambda_%s_opportunity_%s_expansion_%s_%s%s%s.png",
      file_prefix, input_type, lambda_label, alpha_label, beta_label, opportunity_label,
      expansion_label, tree_file_label, sigma_file_suffix, simulation_source_suffix
    )
  )
  if (nzchar(optimal_time_cost_suffix)) {
    pdf_path <- sub("\\.png$", paste0(optimal_time_cost_suffix, ".png"), pdf_path)
  }
  timesteps <- sort(unique(summary_data$decision_timestep))
  if (!is.null(exact_summary) && nrow(exact_summary) > 0) {
    timesteps <- sort(unique(c(timesteps, exact_summary$decision_timestep)))
  }
  if (!is.null(condition_col) && is_disjoint3x2) {
    timesteps <- timesteps[timesteps >= 3L]
  }
  has_exact <- !is.null(exact_summary) && nrow(exact_summary) > 0 && feature_col %in% names(exact_summary)
  condition_levels <- if (!is.null(condition_col)) {
    preferred <- c("incomplete", "complete")
    present <- unique(c(as.character(summary_data[[condition_col]]), if (has_exact) as.character(exact_summary[[condition_col]]) else character()))
    c(preferred[preferred %in% present], setdiff(sort(present), preferred))
  } else {
    "all"
  }
  model_cols_per_condition <- if (has_exact) 2L else 1L
  n_cols <- length(condition_levels) * model_cols_per_condition
  n_rows <- max(1L, length(timesteps) * length(sigma_levels))
  open_panel_png(
    pdf_path,
    n_cols = n_cols,
    n_rows = n_rows,
    legend_fraction = legend_panel_fraction
  )
  old_par <- par(no.readonly = TRUE)
  panel_layout <- cbind(
    matrix(seq_len(n_rows * n_cols), nrow = n_rows, ncol = n_cols, byrow = TRUE),
    rep(n_rows * n_cols + 1L, n_rows)
  )
  layout(panel_layout, widths = c(rep(1, n_cols), legend_panel_fraction))
  par(mar = c(4.2, 4.2, 2.0, 1))
  apply_panel_text_style()
  x_values <- summary_data[[feature_col]]
  y_values <- summary_data$p_continue
  if (has_exact) {
    x_values <- c(x_values, exact_summary[[feature_col]])
    y_values <- c(y_values, exact_summary$p_continue)
  }
  x_limits <- expand_range(x_values, pad = 0.08)
  x_ticks <- seq(floor(x_limits[[1]]), ceiling(x_limits[[2]]), by = 1)
  for (sigma_value in sigma_levels) {
    sigma_title <- if (length(sigma_levels) > 1L) paste0(" | ", sigma_panel_title(sigma_value)) else ""
    sigma_summary <- filter_sigma_rows(summary_data, sigma_value)
    for (decision_timestep in timesteps) {
      for (condition_value in condition_levels) {
        model_piece_all <- sigma_summary[sigma_summary$decision_timestep == decision_timestep, , drop = FALSE]
        if (!is.null(condition_col)) {
          model_piece_all <- model_piece_all[as.character(model_piece_all[[condition_col]]) == condition_value, , drop = FALSE]
        }
        condition_title <- if (!is.null(condition_col)) paste0(" / ", condition_value) else ""
        if (has_exact) {
          exact_piece_all <- exact_summary[exact_summary$decision_timestep == decision_timestep, , drop = FALSE]
          if (!is.null(condition_col) && condition_col %in% names(exact_piece_all)) {
            exact_piece_all <- exact_piece_all[as.character(exact_piece_all[[condition_col]]) == condition_value, , drop = FALSE]
          }
          plot(
            NA,
            xlim = x_limits,
            ylim = c(-0.03, 1.03),
            xlab = xlab,
            ylab = "P(continue)",
            main = sprintf("Optimal%s t%d", condition_title, decision_timestep),
            xaxt = "n"
          )
          axis(1, at = x_ticks)
          grid()
          for (time_cost in sort(unique(exact_summary$time_cost))) {
            piece <- exact_piece_all[
              abs(as.numeric(exact_piece_all$time_cost) - as.numeric(time_cost)) < 1e-8,
              ,
              drop = FALSE
            ]
            if (nrow(piece) == 0) {
              next
            }
            piece <- piece[order(piece[[feature_col]]), , drop = FALSE]
            lines(
              piece[[feature_col]],
              piece$p_continue,
              type = "b",
              pch = 17,
              lty = 1,
              lwd = 2.6,
              col = time_cost_color_for(time_cost)
            )
          }
        }
        plot(
          NA,
          xlim = x_limits,
          ylim = c(-0.03, 1.03),
          xlab = xlab,
          ylab = "P(continue)",
          main = sprintf("VAE%s%s t%d", condition_title, sigma_title, decision_timestep),
          xaxt = "n"
        )
        axis(1, at = x_ticks)
        grid()
        for (opportunity_value in opportunity_levels) {
          for (beta_value in beta_levels) {
            piece <- model_piece_all[
              model_piece_all$opportunity == opportunity_value &
                model_piece_all$beta == beta_value,
              ,
              drop = FALSE
            ]
            if (nrow(piece) == 0) {
              next
            }
            piece <- piece[order(piece[[feature_col]]), , drop = FALSE]
            lines(
              piece[[feature_col]],
              piece$p_continue,
              type = "b",
              pch = opportunity_pch[[as.character(opportunity_value)]],
              lty = opportunity_lty[[as.character(opportunity_value)]],
              lwd = 1.8,
              col = line_color_for(beta_value, opportunity_value)
            )
          }
        }
      }
    }
  }
  draw_legend_panel(include_time_cost = has_exact, cex = 1)
  par(old_par)
  dev.off()
  message(sprintf("Saved %s", pdf_path))
}

condition_action_parameter_label <- function(dat, row_i, source) {
  if (identical(source, "optimal")) {
    return(sprintf("time cost %s", format_plot_values(dat$time_cost[[row_i]])))
  }
  if ("sigma" %in% names(dat) && length(sigma_levels) > 1L) {
    return(sprintf(
      "sigma %s, lambda %s, opportunity %s",
      format_plot_values(dat$sigma[[row_i]]),
      dat$beta[[row_i]],
      dat$opportunity[[row_i]]
    ))
  }
  sprintf("lambda %s, opportunity %s", dat$beta[[row_i]], dat$opportunity[[row_i]])
}

plot_disjoint3x2_condition_action_probabilities <- function(summary_data, exact_summary) {
  if (!is_disjoint3x2) {
    return(invisible(NULL))
  }
  if ((is.null(summary_data) || nrow(summary_data) == 0) &&
      (is.null(exact_summary) || nrow(exact_summary) == 0)) {
    warning("No disjoint3x2 condition/action rows; skipping condition/action plots.")
    return(invisible(NULL))
  }

  plot_data <- list()
  if (!is.null(exact_summary) && nrow(exact_summary) > 0) {
    exact_piece <- exact_summary
    exact_piece$source <- "optimal"
    exact_piece$parameter_label <- vapply(
      seq_len(nrow(exact_piece)),
      function(i) condition_action_parameter_label(exact_piece, i, "optimal"),
      character(1)
    )
    plot_data[[length(plot_data) + 1L]] <- exact_piece
  }
  if (!is.null(summary_data) && nrow(summary_data) > 0) {
    model_piece <- summary_data
    model_piece$source <- "vae"
    model_piece$time_cost <- NA_real_
    model_piece$parameter_label <- vapply(
      seq_len(nrow(model_piece)),
      function(i) condition_action_parameter_label(model_piece, i, "vae"),
      character(1)
    )
    plot_data[[length(plot_data) + 1L]] <- model_piece
  }
  if (length(plot_data) == 0) {
    return(invisible(NULL))
  }
  dat <- bind_rows_fill(plot_data)
  required_cols <- c(
    "source",
    "parameter_label",
    "decision_timestep",
    "count_pattern",
    "best_path_identity",
    "best_path_value",
    "action_label",
    "p_action"
  )
  if (!all(required_cols %in% names(dat))) {
    warning("Condition/action summary is missing required columns; skipping condition/action plots.")
    return(invisible(NULL))
  }
  dat$decision_timestep <- suppressWarnings(as.integer(dat$decision_timestep))
  dat$best_path_value <- suppressWarnings(as.numeric(dat$best_path_value))
  dat$p_action <- suppressWarnings(as.numeric(dat$p_action))
  dat <- dat[
    !is.na(dat$decision_timestep) &
      is.finite(dat$best_path_value) &
      is.finite(dat$p_action),
    ,
    drop = FALSE
  ]
  if (nrow(dat) == 0) {
    return(invisible(NULL))
  }

  action_levels <- c(
    "observe_P1", "observe_P2", "observe_P3",
    "stop_P1", "stop_P2", "stop_P3", "stop"
  )
  present_actions <- unique(as.character(dat$action_label))
  action_levels <- c(action_levels[action_levels %in% present_actions], setdiff(sort(present_actions), action_levels))
  action_cols <- setNames(
    c("#1b9e77", "#66a61e", "#a6d854", "#d95f02", "#e7298a", "#e6ab02", "#666666")[seq_along(action_levels)],
    action_levels
  )
  action_lty <- setNames(rep(1, length(action_levels)), action_levels)
  action_pch <- setNames(rep(16, length(action_levels)), action_levels)
  if ("stop" %in% action_levels) {
    action_lty[["stop"]] <- 2
    action_pch[["stop"]] <- 17
  }
  stop_actions <- intersect(c("stop_P1", "stop_P2", "stop_P3"), action_levels)
  if (length(stop_actions) > 0) {
    action_pch[stop_actions] <- 17
  }

  source_order <- c("optimal", "vae")
  best_order <- c("P1", "not_P1", "P2", "P3", "tie")
  timesteps <- sort(unique(dat$decision_timestep))
  timesteps <- timesteps[timesteps >= 2L & timesteps <= task_node_count]
  for (decision_timestep in timesteps) {
    timestep_data <- dat[dat$decision_timestep == decision_timestep, , drop = FALSE]
    if (nrow(timestep_data) == 0) {
      next
    }
    timestep_data$source <- factor(timestep_data$source, levels = source_order)
    timestep_data$best_path_identity <- factor(timestep_data$best_path_identity, levels = best_order)
    panel_keys <- unique(timestep_data[, c(
      "source", "parameter_label", "count_pattern", "best_path_identity"
    ), drop = FALSE])
    panel_keys <- panel_keys[order(
      panel_keys$source,
      panel_keys$parameter_label,
      panel_keys$count_pattern,
      panel_keys$best_path_identity
    ), , drop = FALSE]
    if (nrow(panel_keys) == 0) {
      next
    }
    n_cols <- min(4L, nrow(panel_keys))
    n_rows <- ceiling(nrow(panel_keys) / n_cols)
    pdf_path <- file.path(
      results_dir,
      sprintf(
        "disjoint3x2_condition_action_probabilities_t%d_%s_loss_scale_%s_alpha_%s_lambda_%s_opportunity_%s_expansion_%s_%s%s%s.png",
        decision_timestep,
        input_type, lambda_label, alpha_label, beta_label, opportunity_label,
        expansion_label, tree_file_label, sigma_file_suffix, simulation_source_suffix
      )
    )
    if (nzchar(optimal_time_cost_suffix)) {
      pdf_path <- sub("\\.png$", paste0(optimal_time_cost_suffix, ".png"), pdf_path)
    }
    open_panel_png(
      pdf_path,
      n_cols = n_cols,
      n_rows = n_rows,
      legend_fraction = legend_panel_fraction
    )
    old_par <- par(no.readonly = TRUE)
    panel_layout <- cbind(
      matrix(seq_len(n_rows * n_cols), nrow = n_rows, ncol = n_cols, byrow = TRUE),
      rep(n_rows * n_cols + 1L, n_rows)
    )
    layout(panel_layout, widths = c(rep(1, n_cols), legend_panel_fraction))
    par(mar = c(4.2, 4.2, 2.2, 1))
    apply_panel_text_style()
    x_limits <- expand_range(timestep_data$best_path_value, pad = 0.08)
    x_ticks <- seq(floor(x_limits[[1]]), ceiling(x_limits[[2]]), by = 1)

    for (panel_i in seq_len(n_rows * n_cols)) {
      if (panel_i > nrow(panel_keys)) {
        plot.new()
        next
      }
      key <- panel_keys[panel_i, , drop = FALSE]
      panel_data <- timestep_data[
        as.character(timestep_data$source) == as.character(key$source[[1]]) &
          timestep_data$parameter_label == key$parameter_label[[1]] &
          timestep_data$count_pattern == key$count_pattern[[1]] &
          as.character(timestep_data$best_path_identity) == as.character(key$best_path_identity[[1]]),
        ,
        drop = FALSE
      ]
      plot(
        NA,
        xlim = x_limits,
        ylim = c(0, 1),
        xlab = "Best path value from actual observed rewards",
        ylab = "P(action)",
        main = sprintf(
          "%s | %s\n%s, best=%s",
          as.character(key$source[[1]]),
          key$parameter_label[[1]],
          key$count_pattern[[1]],
          as.character(key$best_path_identity[[1]])
        ),
        xaxt = "n",
        cex.main = 0.8
      )
      axis(1, at = x_ticks)
      grid()
      for (action_label in action_levels) {
        piece <- panel_data[as.character(panel_data$action_label) == action_label, , drop = FALSE]
        if (nrow(piece) == 0) {
          next
        }
        piece <- piece[order(piece$best_path_value), , drop = FALSE]
        lines(
          piece$best_path_value,
          piece$p_action,
          type = "b",
          col = action_cols[[action_label]],
          lty = action_lty[[action_label]],
          pch = action_pch[[action_label]],
          lwd = 1.5
        )
      }
    }
    par(mar = c(0, 0, 0, 0))
    plot.new()
    legend(
      "center",
      legend = action_levels,
      col = unname(action_cols[action_levels]),
      lty = unname(action_lty[action_levels]),
      pch = unname(action_pch[action_levels]),
      lwd = 1.5,
      bty = "n",
      cex = 0.9
    )
    par(old_par)
    dev.off()
    message(sprintf("Saved %s", pdf_path))
  }
}

build_kl_by_best_observed_continue_summary <- function(dat) {
  use_signed_path_difference <- is_default2 || is_disjoint2x2
  kl_payment_offset <- if (expansion_decision_version %in% c("lstm", "pre_lstm")) 1L else 0L
  reward_timesteps <- column_timesteps(dat, "^expanded_reward_t[0-9]+$", "^expanded_reward_t")
  node_timesteps <- column_timesteps(dat, "^expanded_node_t[0-9]+$", "^expanded_node_t")
  stop_timesteps <- column_timesteps(dat, "^stop_t[0-9]+$", "^stop_t")
  kl_timesteps <- column_timesteps(dat, "^kl_d_t[0-9]+$", "^kl_d_t")
  observation_timesteps <- Reduce(
    intersect,
    list(reward_timesteps, node_timesteps, kl_timesteps)
  )
  observation_timesteps <- observation_timesteps[
    (observation_timesteps + 1L) %in% stop_timesteps &
      (observation_timesteps + kl_payment_offset) %in% kl_timesteps
  ]
  if (length(observation_timesteps) == 0) {
    return(data.frame())
  }
  required_cols <- unique(c(
    paste0("expanded_node_t", observation_timesteps),
    paste0("expanded_reward_t", observation_timesteps),
    paste0("stop_t", unique(c(observation_timesteps, observation_timesteps + 1L))),
    paste0("kl_d_t", observation_timesteps + kl_payment_offset)
  ))
  required_cols <- required_cols[required_cols %in% names(dat)]
  trial_data <- unique_trial_rows(dat, required_cols)
  if (nrow(trial_data) == 0) {
    return(data.frame())
  }
  actual_lookup <- build_node_actual_reward_lookup(dat)
  rows <- list()
  row_index <- seq_len(nrow(trial_data))
  path_values <- matrix(0, nrow = nrow(trial_data), ncol = task_path_count)
  path_counts <- matrix(0L, nrow = nrow(trial_data), ncol = task_path_count)

  for (observation_timestep in observation_timesteps) {
    node_col <- paste0("expanded_node_t", observation_timestep)
    reward_col <- paste0("expanded_reward_t", observation_timestep)
    continue_stop_col <- paste0("stop_t", observation_timestep + 1L)
    kl_payment_timestep <- observation_timestep + kl_payment_offset
    kl_col <- paste0("kl_d_t", kl_payment_timestep)
    if (!all(c(node_col, reward_col, continue_stop_col, kl_col) %in% names(trial_data))) {
      next
    }

    node_values <- suppressWarnings(as.integer(trial_data[[node_col]]))
    reward_values_t <- observed_actual_reward_values(trial_data, node_col, reward_col, actual_lookup)
    path_indices <- path_id_for_node(node_values)
    alive_before_observation <- row_alive_before_decision(trial_data, observation_timestep + 1L)
    valid_observation <- alive_before_observation &
      !is.na(path_indices) &
      !is.na(reward_values_t) &
      path_indices >= 1L &
      path_indices <= task_path_count
    if (any(valid_observation)) {
      idx <- cbind(row_index[valid_observation], path_indices[valid_observation])
      path_values[idx] <- path_values[idx] + reward_values_t[valid_observation]
      path_counts[idx] <- path_counts[idx] + 1L
    }

    stop_values <- trial_data[[continue_stop_col]]
    kl_values <- suppressWarnings(as.numeric(trial_data[[kl_col]]))
    observed_path_values <- path_values
    observed_path_values[path_counts <= 0L] <- -Inf
    best_path_values <- apply(observed_path_values, 1L, max)
    best_path_values[!is.finite(best_path_values)] <- NA_real_
    if (use_signed_path_difference) {
      other_path_indices <- ifelse(path_indices == 1L, 2L, 1L)
      x_values <- rep(NA_real_, nrow(trial_data))
      valid_difference <- valid_observation &
        !is.na(other_path_indices) &
        path_indices >= 1L &
        path_indices <= 2L
      if (any(valid_difference)) {
        x_values[valid_difference] <- path_values[cbind(row_index[valid_difference], path_indices[valid_difference])] -
          path_values[cbind(row_index[valid_difference], other_path_indices[valid_difference])]
      }
    } else {
      x_values <- best_path_values
    }

    keep <- valid_observation &
      !is.na(stop_values) &
      !as_logical_col(stop_values) &
      is.finite(kl_values) &
      is.finite(x_values)
    if (!any(keep)) {
      next
    }

    rows[[length(rows) + 1L]] <- data.frame(
      sigma = trial_data$sigma[keep],
      beta = trial_data$beta[keep],
      opportunity = trial_data$opportunity[keep],
      observation_timestep = observation_timestep,
      continue_decision_timestep = observation_timestep + 1L,
      kl_payment_timestep = kl_payment_timestep,
      x_value = x_values[keep],
      best_path_value = best_path_values[keep],
      kl_paid = kl_values[keep],
      stringsAsFactors = FALSE
    )
  }

  if (length(rows) == 0) {
    return(data.frame())
  }
  kl_data <- do.call(rbind, rows)
  summary_data <- aggregate(
    kl_paid ~ sigma + beta + opportunity + observation_timestep + continue_decision_timestep + kl_payment_timestep + x_value,
    data = kl_data,
    FUN = mean
  )
  count_data <- aggregate(
    kl_paid ~ sigma + beta + opportunity + observation_timestep + continue_decision_timestep + kl_payment_timestep + x_value,
    data = kl_data,
    FUN = length
  )
  names(count_data)[names(count_data) == "kl_paid"] <- "n"
  out <- merge(
    summary_data,
    count_data,
    by = c("sigma", "beta", "opportunity", "observation_timestep", "continue_decision_timestep", "kl_payment_timestep", "x_value")
  )
  out$x_value_type <- if (use_signed_path_difference) "signed_path_difference" else "best_observed_path_value"
  out
}

plot_kl_by_best_observed_continue_summary <- function(summary_data) {
  use_signed_path_difference <- isTRUE(any(summary_data$x_value_type == "signed_path_difference"))
  file_prefix <- if (use_signed_path_difference) {
    "kl_paid_by_observed_path_value_difference_after_continue"
  } else {
    "kl_paid_by_best_observed_path_value_after_continue"
  }
  if (nrow(summary_data) == 0) {
    warning(sprintf("No model rows for %s; skipping plot.", file_prefix))
    return(invisible(NULL))
  }
  pdf_path <- file.path(
    results_dir,
    sprintf(
      "%s_%s_loss_scale_%s_alpha_%s_lambda_%s_opportunity_%s_expansion_%s_%s%s%s.png",
      file_prefix, input_type, lambda_label, alpha_label, beta_label, opportunity_label,
      expansion_label, tree_file_label, sigma_file_suffix, simulation_source_suffix
    )
  )
  timesteps <- sort(unique(summary_data$observation_timestep))
  n_rows <- max(1L, length(timesteps))
  n_sigma <- max(1L, length(sigma_levels))
  open_panel_png(
    pdf_path,
    n_cols = n_sigma,
    n_rows = n_rows
  )
  old_par <- par(no.readonly = TRUE)
  par(mfrow = c(n_rows, n_sigma))
  par(mar = c(4.2, 4.2, if (n_sigma > 1L) 2.2 else 2.0, 1))
  apply_panel_text_style()
  x_limits <- expand_range(summary_data$x_value, pad = 0.08)
  y_limits <- expand_range(summary_data$kl_paid, pad = 0.08)
  x_ticks <- seq(floor(x_limits[[1]]), ceiling(x_limits[[2]]), by = 1)
  x_label <- if (use_signed_path_difference) {
    "Current path value - other path value (actual rewards)"
  } else {
    "Best path value from actual observed rewards"
  }
  for (observation_timestep in timesteps) {
    payment_timesteps <- sort(unique(summary_data$kl_payment_timestep[
      summary_data$observation_timestep == observation_timestep
    ]))
    payment_label <- if (length(payment_timesteps) == 1L && is.finite(payment_timesteps[[1]])) {
      sprintf("paid KL t%d", as.integer(payment_timesteps[[1]]))
    } else {
      "paid KL"
    }
    for (sigma_value in sigma_levels) {
      sigma_summary <- filter_sigma_rows(summary_data, sigma_value)
      plot(
        NA,
        xlim = x_limits,
        ylim = y_limits,
        xlab = x_label,
        ylab = "Mean KL paid at timestep",
        main = sprintf(
          "VAE obs t%d, continued at decision t%d, %s%s",
          observation_timestep,
          observation_timestep + 1L,
          payment_label,
          if (n_sigma > 1L) paste0("\n", sigma_panel_title(sigma_value)) else ""
        ),
        xaxt = "n"
      )
      axis(1, at = x_ticks)
      grid()
      for (opportunity_value in opportunity_levels) {
        for (beta_value in beta_levels) {
          piece <- sigma_summary[
            sigma_summary$observation_timestep == observation_timestep &
              sigma_summary$opportunity == opportunity_value &
              sigma_summary$beta == beta_value,
            ,
            drop = FALSE
          ]
          if (nrow(piece) == 0) {
            next
          }
          piece <- piece[order(piece$x_value), , drop = FALSE]
          lines(
            piece$x_value,
            piece$kl_paid,
            type = "b",
            pch = opportunity_pch[[as.character(opportunity_value)]],
            lty = opportunity_lty[[as.character(opportunity_value)]],
            lwd = 1.8,
            col = line_color_for(beta_value, opportunity_value)
          )
        }
      }
    }
  }
  par(old_par)
  dev.off()
  message(sprintf("Saved %s", pdf_path))
}

if (is_bandit3 || is_bandit4 || is_disjoint3x2) {
  continue_summary <- build_continue_feature_summary(all_data, feature_type = "best")
  exact_continue_summary <- build_exact_continue_summary(feature_type = "best")
  plot_continue_summary(
    continue_summary,
    exact_continue_summary,
    feature_col = "best_path_value",
    xlab = "Best path value from actual observed rewards",
    file_prefix = "p_continue_by_best_observed_path_value"
  )
} else if (is_default2 || is_disjoint2x2) {
  continue_summary <- build_continue_feature_summary(all_data, feature_type = "difference")
  exact_continue_summary <- build_exact_continue_summary(feature_type = "difference")
  plot_continue_summary(
    continue_summary,
    exact_continue_summary,
    feature_col = "path_value_difference",
    xlab = "Current observed path value - other path value (actual rewards)",
    file_prefix = "p_continue_by_observed_path_value_difference"
  )
} else {
  warning(sprintf(
    "Tree config %s with tree_size=%s is not one of default2, bandit3, bandit4, disjoint2x2, or disjoint3x2; continue-policy plot skipped.",
    tree_config,
    tree_size
  ))
}

if (is_disjoint3x2) {
  condition_action_summary <- build_disjoint3x2_condition_action_summary(all_data)
  exact_condition_action_summary <- build_exact_disjoint3x2_condition_action_summary()
  plot_disjoint3x2_condition_action_probabilities(
    condition_action_summary,
    exact_condition_action_summary
  )
}

if (identical(model_variant, "vae")) {
  kl_by_best_summary <- build_kl_by_best_observed_continue_summary(all_data)
  plot_kl_by_best_observed_continue_summary(kl_by_best_summary)
}
