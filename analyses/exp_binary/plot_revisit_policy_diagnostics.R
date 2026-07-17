#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)

script_file_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
script_dir <- if (length(script_file_arg) > 0L) {
  dirname(normalizePath(sub("^--file=", "", script_file_arg[[1]]), mustWork = FALSE))
} else {
  "analyses/exp_binary"
}

trim_string <- function(value) {
  trimws(as.character(value))
}

extract_named_option <- function(args, option_names, default = NULL) {
  value <- default
  keep <- rep(TRUE, length(args))
  i <- 1L
  while (i <= length(args)) {
    arg <- args[[i]]
    matched_name <- NA_character_
    matched_inline <- FALSE
    for (option_name in option_names) {
      inline_prefix <- paste0(option_name, "=")
      if (identical(arg, option_name)) {
        matched_name <- option_name
        break
      }
      if (startsWith(arg, inline_prefix)) {
        matched_name <- option_name
        matched_inline <- TRUE
        break
      }
    }
    if (is.na(matched_name)) {
      i <- i + 1L
      next
    }
    if (isTRUE(matched_inline)) {
      value <- sub(paste0("^", matched_name, "="), "", arg)
      keep[[i]] <- FALSE
      i <- i + 1L
    } else {
      if (i == length(args)) {
        stop(sprintf("%s requires a value.", matched_name))
      }
      value <- args[[i + 1L]]
      keep[[i]] <- FALSE
      keep[[i + 1L]] <- FALSE
      i <- i + 2L
    }
  }
  list(args = args[keep], value = value)
}

extract_flag_option <- function(args, option_names) {
  keep <- rep(TRUE, length(args))
  found <- FALSE
  for (i in seq_along(args)) {
    if (args[[i]] %in% option_names) {
      keep[[i]] <- FALSE
      found <- TRUE
    }
  }
  list(args = args[keep], found = found)
}

parse_nonnegative_integer_option <- function(value, option_name) {
  numeric_value <- suppressWarnings(as.numeric(value))
  if (
    is.na(numeric_value) ||
      !is.finite(numeric_value) ||
      numeric_value < 0 ||
      abs(numeric_value - round(numeric_value)) > 1e-9
  ) {
    stop(sprintf("%s must be a nonnegative integer. Got %s.", option_name, value))
  }
  as.integer(round(numeric_value))
}

parse_positive_numeric_option <- function(value, option_name) {
  numeric_value <- suppressWarnings(as.numeric(value))
  if (
    is.na(numeric_value) ||
      !is.finite(numeric_value) ||
      numeric_value <= 0
  ) {
    stop(sprintf("%s must be a positive number. Got %s.", option_name, value))
  }
  numeric_value
}

normalize_sampled_lambda_critic <- function(value) {
  key <- tolower(trim_string(value))
  aliases <- c(
    "q" = "q", "action_q" = "q", "action-q" = "q", "qcritic" = "q",
    "legacy" = "q", "old" = "q", "none" = "q",
    "v" = "value", "value" = "value", "scalar" = "value",
    "scalar_v" = "value", "scalar-v" = "value", "vcritic" = "value"
  )
  if (!key %in% names(aliases)) {
    stop(sprintf(
      "--sampled-lambda-critic must be q or value/v. Got %s.",
      value
    ))
  }
  unname(aliases[[key]])
}

min_samples_option <- extract_named_option(
  args,
  c("--min-samples", "--min-sampes", "--minimum-samples", "--min-sample-count", "--min-n"),
  default = "10"
)
args <- min_samples_option$args
minimum_samples_threshold <- parse_nonnegative_integer_option(
  min_samples_option$value,
  "--min-samples"
)
log_axis_floor_option <- extract_named_option(
  args,
  c("--log-axis-floor", "--log-floor", "--kl-log-floor"),
  default = "1e-8"
)
args <- log_axis_floor_option$args
log_axis_floor <- parse_positive_numeric_option(
  log_axis_floor_option$value,
  "--log-axis-floor"
)
selected_revisit_plots_option <- extract_flag_option(
  args,
  c("--selected-plots-only", "--core-plots-only", "--focused-plots-only", "--kl-entropy-core-only")
)
args <- selected_revisit_plots_option$args
selected_revisit_plots_only <- selected_revisit_plots_option$found
node_coverage_aux_coef_option <- extract_named_option(
  args,
  c("--node-coverage-aux-coef", "--aux-coef"),
  default = NULL
)
args <- node_coverage_aux_coef_option$args
node_coverage_aux_epochs_option <- extract_named_option(
  args,
  c("--node-coverage-aux-epochs", "--aux-epochs"),
  default = NULL
)
args <- node_coverage_aux_epochs_option$args
critic_option <- extract_named_option(
  args,
  c("--sampled-lambda-critic", "--critic", "--critic-type", "--critic-mode"),
  default = "q"
)
args <- critic_option$args
sampled_lambda_critic <- normalize_sampled_lambda_critic(critic_option$value)

normalize_preset_tree <- function(value) {
  key <- tolower(trim_string(value))
  aliases <- c(
    "2" = "default", "2n" = "default", "default" = "default", "default2" = "default",
    "3" = "bandit3", "3n" = "bandit3", "bandit3" = "bandit3",
    "4" = "disjoint2x2", "4n" = "disjoint2x2", "disjoint2x2" = "disjoint2x2", "2x2" = "disjoint2x2",
    "6" = "disjoint3x2", "6n" = "disjoint3x2", "disjoint3x2" = "disjoint3x2", "3x2" = "disjoint3x2"
  )
  if (key %in% names(aliases)) {
    return(unname(aliases[[key]]))
  }
  key
}

normalize_preset_vary <- function(value) {
  key <- tolower(trim_string(value))
  aliases <- c(
    "beta" = "beta", "betas" = "beta", "memory" = "beta", "memory_cost" = "beta",
    "lambda" = "beta", "lambdas" = "beta", "memory_lambda" = "beta", "memory-lambda" = "beta",
    "opportunity" = "opportunity", "opp" = "opportunity", "opportunity_cost" = "opportunity",
    "gamma" = "opportunity", "time" = "opportunity", "time_cost" = "opportunity"
  )
  if (key %in% names(aliases)) {
    return(unname(aliases[[key]]))
  }
  key
}

revisit_preset_file <- file.path(script_dir, "revisit_plot_presets.csv")
revisit_preset_node_coverage_aux_coef <- NULL
revisit_preset_node_coverage_aux_epochs <- NULL
preset_arg_column_specs <- list(
  c("memory_lambda_arg", "beta_arg"),
  c("loss_scale_arg", "lambda_arg"),
  c("alpha_arg"),
  c("opportunity_arg"),
  c("input_dir"),
  c("results_dir"),
  c("tree_size"),
  c("input_type"),
  c("expansion_decision_version"),
  c("model_variant"),
  c("tree_config"),
  c("seed_arg"),
  c("rnn_units_arg"),
  c("latent_dim_arg"),
  c("simulation_source_arg"),
  c("max_observations_arg"),
  c("sigma_arg"),
  c("optimal_dir"),
  c("optimal_opportunity_arg")
)

read_revisit_presets <- function(path) {
  if (!file.exists(path)) {
    return(data.frame())
  }
  presets <- read.csv(path, stringsAsFactors = FALSE, check.names = FALSE)
  missing_columns <- c(setdiff(c("tree", "vary"), names(presets)), vapply(
    preset_arg_column_specs,
    function(spec) {
      if (any(spec %in% names(presets))) "" else paste(spec, collapse = " or ")
    },
    character(1)
  ))
  missing_columns <- missing_columns[nzchar(missing_columns)]
  if (length(missing_columns) > 0L) {
    stop(sprintf(
      "Preset file %s is missing required column(s): %s",
      path,
      paste(missing_columns, collapse = ", ")
    ))
  }
  presets$tree_key <- vapply(presets$tree, normalize_preset_tree, character(1))
  presets$vary_key <- vapply(presets$vary, normalize_preset_vary, character(1))
  presets
}

print_revisit_preset_usage <- function(presets) {
  message("Preset usage:")
  message("  Rscript analyses/exp_binary/plot_revisit_policy_diagnostics.R <tree> <beta|opportunity>")
  message("Examples:")
  message("  Rscript analyses/exp_binary/plot_revisit_policy_diagnostics.R default beta")
  message("  Rscript analyses/exp_binary/plot_revisit_policy_diagnostics.R bandit3 opportunity")
  message("  Rscript analyses/exp_binary/plot_revisit_policy_diagnostics.R disjoint3x2 opportunity")
  if (nrow(presets) > 0L) {
    message("Available presets:")
    for (i in seq_len(nrow(presets))) {
      message(sprintf("  %s %s", presets$tree_key[[i]], presets$vary_key[[i]]))
    }
  }
}

apply_revisit_preset_args <- function(args) {
  presets <- read_revisit_presets(revisit_preset_file)
  if (length(args) >= 1L && tolower(args[[1]]) %in% c("--list-presets", "list-presets", "presets")) {
    print_revisit_preset_usage(presets)
    quit(save = "no", status = 0)
  }

  preset_offset <- 0L
  if (length(args) >= 3L && tolower(args[[1]]) %in% c("--preset", "preset")) {
    preset_offset <- 1L
  } else if (length(args) >= 2L) {
    first_numeric <- !is.na(suppressWarnings(as.numeric(args[[1]])))
    second_vary <- normalize_preset_vary(args[[2]]) %in% c("beta", "opportunity")
    if (!first_numeric && second_vary) {
      preset_offset <- 0L
    }
  }

  if (preset_offset < 0L || preset_offset == 0L && !(length(args) >= 2L && !is.na(match(normalize_preset_vary(args[[2]]), c("beta", "opportunity"))))) {
    return(args)
  }
  if (preset_offset == 1L && length(args) < 3L) {
    print_revisit_preset_usage(presets)
    stop("--preset requires <tree> and <beta|opportunity>.")
  }

  tree_arg <- args[[preset_offset + 1L]]
  vary_arg <- args[[preset_offset + 2L]]
  tree_key <- normalize_preset_tree(tree_arg)
  vary_key <- normalize_preset_vary(vary_arg)

  if (nrow(presets) == 0L) {
    stop(sprintf("No revisit preset file found at %s.", revisit_preset_file))
  }

  match_idx <- which(presets$tree_key == tree_key & presets$vary_key == vary_key)
  if (length(match_idx) != 1L) {
    print_revisit_preset_usage(presets)
    stop(sprintf("No unique revisit preset found for tree=%s vary=%s.", tree_arg, vary_arg))
  }

  preset <- presets[match_idx[[1]], , drop = FALSE]
  optional_preset_value <- function(column, default) {
    if (!column %in% names(preset)) {
      return(default)
    }
    value <- preset[[column]][[1]]
    if (is.na(value) || !nzchar(trim_string(value))) default else as.character(value)
  }
  revisit_preset_node_coverage_aux_coef <<- optional_preset_value("node_coverage_aux_coef_arg", "0")
  revisit_preset_node_coverage_aux_epochs <<- optional_preset_value("node_coverage_aux_epochs_arg", "0")
  preset_args <- unname(vapply(preset_arg_column_specs, function(spec) {
    col <- spec[spec %in% names(preset)][[1]]
    value <- preset[[col]][[1]]
    if (is.na(value)) "" else as.character(value)
  }, character(1)))

  extra_args <- if (length(args) > preset_offset + 2L) {
    args[(preset_offset + 3L):length(args)]
  } else {
    character()
  }
  if (length(extra_args) > 0L) {
    stop(sprintf(
      "Preset mode only accepts <tree> <beta|opportunity> right now. Unexpected extra argument(s): %s",
      paste(extra_args, collapse = " ")
    ))
  }

  message(sprintf(
    "Using revisit plot preset: tree=%s vary=%s from %s",
    tree_key,
    vary_key,
    revisit_preset_file
  ))
  preset_args
}

args <- apply_revisit_preset_args(args)

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
node_coverage_aux_coef_arg <- if (!is.null(node_coverage_aux_coef_option$value)) {
  trim_string(node_coverage_aux_coef_option$value)
} else if (!is.null(revisit_preset_node_coverage_aux_coef)) {
  trim_string(revisit_preset_node_coverage_aux_coef)
} else {
  "0"
}
node_coverage_aux_epochs_arg <- if (!is.null(node_coverage_aux_epochs_option$value)) {
  trim_string(node_coverage_aux_epochs_option$value)
} else if (!is.null(revisit_preset_node_coverage_aux_epochs)) {
  trim_string(revisit_preset_node_coverage_aux_epochs)
} else {
  "0"
}

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
message(sprintf("Minimum samples per plotted dot or heatmap cell: %d", minimum_samples_threshold))
message(sprintf("Log-axis display floor: %g", log_axis_floor))
message(sprintf(
  "Sampled-lambda critic file mode: %s%s",
  sampled_lambda_critic,
  if (identical(sampled_lambda_critic, "q")) " (legacy/no _vcritic suffix)" else " (_vcritic suffix)"
))
message(sprintf(
  "Node coverage aux file mode: coef=%s epochs=%s",
  node_coverage_aux_coef_arg,
  node_coverage_aux_epochs_arg
))
if (isTRUE(selected_revisit_plots_only)) {
  message("Selected revisit plot mode: writing only the core KL/entropy/reward plots.")
}

tree_file_label <- paste0(tree_size, "n", if (nzchar(tree_config)) paste0("_", tree_config) else "")
architecture_file_label <- sprintf("rnn_%s_latent_%s", rnn_units_arg, latent_dim_arg)
revisit_label <- paste0("revisit_maxobs_", max_observations_arg)
source_suffix <- if (identical(simulation_source, "jax")) "_source_jax" else ""
critic_output_suffix <- if (identical(sampled_lambda_critic, "value")) "_vcritic" else ""

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
  "%s_lambda_%s_alpha_%s_beta_%s_opportunity_%s_expansion_%s_variant_%s_%s_%s_%s%s%s%s%s",
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
  source_suffix,
  critic_output_suffix
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

selected_revisit_axis_plot_prefixes <- c(
  "kl_paid_at_first_timestep_after_continue_vs_first_observed_reward_t1_sigma_panels",
  "kl_paid_at_stop_timestep_after_first_continue_vs_absolute_first_observed_minus_mean_other_path_sigma_panels",
  "kl_paid_at_timestep_vs_timestep_sigma_panels",
  "kl_paid_at_timestep_vs_timestep_by_total_timestep_sigma_panels",
  "terminal_binary_choice_entropy_at_timestep_vs_strict_pre_stop_timestep_sigma_panels",
  "timestep_before_stop_vs_normalized_chosen_path_reward_sigma_panels",
  "kl_paid_total_vs_normalized_chosen_path_reward_sigma_panels"
)

should_write_revisit_axis_plot <- function(file_prefix) {
  !isTRUE(selected_revisit_plots_only) ||
    axis_filename_label(file_prefix) %in% selected_revisit_axis_plot_prefixes
}

safe_png_path <- function(file_prefix, suffix = NULL, max_basename_chars = 180) {
  clean_prefix <- axis_filename_label(file_prefix)
  basename <- sprintf("%s.png", clean_prefix)
  if (nchar(basename, type = "bytes") <= max_basename_chars) {
    return(file.path(plot_output_dir, basename))
  }
  hash <- short_string_hash(basename)
  fixed_chars <- nchar(hash, type = "bytes") + nchar("_h.png", type = "bytes")
  keep_chars <- max(16L, max_basename_chars - fixed_chars)
  file.path(plot_output_dir, sprintf("%s_h%s.png", substr(clean_prefix, 1L, keep_chars), hash))
}

plot_font_size_pt <- 7
panel_plot_width_in <- 33 / 25.4
panel_plot_height_in <- 33 / 25.4
panel_margin_line_height_in <- plot_font_size_pt * 1.2 / 72
legend_panel_fraction <- 0.75
panel_left_margin_lines <- 6.8
panel_right_gap_lines <- 4.0
panel_bottom_margin_lines <- 5.2
panel_top_margin_lines <- 1.4
panel_title_margin_lines <- 2.8

panel_margins <- function(top = panel_top_margin_lines, bottom = panel_bottom_margin_lines) {
  c(bottom, panel_left_margin_lines, top, panel_right_gap_lines)
}

panel_cell_width_in <- function() {
  panel_plot_width_in + (panel_left_margin_lines + panel_right_gap_lines) * panel_margin_line_height_in
}

panel_cell_height_in <- function(top = panel_top_margin_lines, bottom = panel_bottom_margin_lines) {
  panel_plot_height_in + (top + bottom) * panel_margin_line_height_in
}

open_panel_png <- function(path, n_cols = 1L, n_rows = 1L, legend_fraction = 0) {
  cell_width <- panel_cell_width_in()
  cell_height <- panel_cell_height_in()
  png(
    path,
    width = cell_width * n_cols + panel_plot_width_in * legend_fraction,
    height = cell_height * n_rows,
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

filename_has_value_critic <- function(path) {
  grepl("_vcritic(_|\\.)", basename(path), perl = TRUE)
}

critic_file_matches <- function(path) {
  has_value_critic <- filename_has_value_critic(path)
  if (identical(sampled_lambda_critic, "value")) {
    return(has_value_critic)
  }
  !has_value_critic
}

sampled_lambda_critic_file_suffixes <- function() {
  if (identical(sampled_lambda_critic, "value")) "_vcritic" else c("", "_qcritic")
}

stop_paid_suffix_variants <- function(suffixes) {
  unique(c(suffixes, paste0(suffixes, "_stop_paid")))
}

node_coverage_suffix_variants <- function(suffixes) {
  coef_num <- suppressWarnings(as.numeric(node_coverage_aux_coef_arg))
  if (!is.finite(coef_num) || abs(coef_num) < 1e-12) {
    return(unique(suffixes))
  }
  nodecov_suffixes <- as.vector(outer(
    value_candidates(node_coverage_aux_coef_arg),
    value_candidates(node_coverage_aux_epochs_arg),
    function(coef_token, epochs_token) paste0("_nodecov_", coef_token, "_anneal_", epochs_token)
  ))
  unique(as.vector(outer(suffixes, nodecov_suffixes, paste0)))
}

filename_node_coverage_aux_values <- function(path) {
  matches <- regexec("_nodecov_([^_]+)_anneal_([^_]+)", basename(path), perl = TRUE)
  pieces <- regmatches(basename(path), matches)[[1]]
  if (length(pieces) < 3L) {
    return(c(coef = 0, epochs = 0))
  }
  values <- suppressWarnings(as.numeric(pieces[2:3]))
  values[!is.finite(values)] <- 0
  stats::setNames(values, c("coef", "epochs"))
}

node_coverage_aux_file_matches <- function(path) {
  requested <- suppressWarnings(as.numeric(c(node_coverage_aux_coef_arg, node_coverage_aux_epochs_arg)))
  requested[!is.finite(requested)] <- 0
  found <- filename_node_coverage_aux_values(path)
  all(abs(found - requested) < 1e-8)
}

visited_lstm_suffix_variants <- function(suffixes) {
  unique(c(suffixes, paste0(suffixes, "_visitedidx")))
}

revisit_optional_suffix_regex <- function() {
  paste0(
    "(_obs_sigma_[^_]+)?",
    "(_klstart_[^_]+_klanneal_[^_]+)?",
    "(_nodecov_[^_]+_anneal_[^_]+)?",
    "(_(?:q|v)critic)?",
    "(_stop_paid)?",
    "(_observer_endchoice)?",
    "(_visitedidx)?"
  )
}

numeric_file_match <- function(lambda_value, alpha_value, beta_value, opportunity_value, seed, sigma_value) {
  requested <- suppressWarnings(as.numeric(c(lambda_value, alpha_value, beta_value, opportunity_value)))
  requested_maxobs <- suppressWarnings(as.numeric(max_observations_arg))
  if (any(is.na(requested)) || is.na(requested_maxobs)) {
    return(NA_character_)
  }
  files <- list.files(input_dir, full.names = TRUE)
  files <- files[vapply(files, critic_file_matches, logical(1))]
  files <- files[vapply(files, node_coverage_aux_file_matches, logical(1))]
  for (tree_label_candidate in simulation_tree_file_labels()) {
    for (variant_file_segment in variant_file_segments) {
      patterns <- c(
        paste0(
          "^loss_scale_([^_]+)_alpha_([^_]+)_lambda_([^_]+)_opportunity_([^_]+)_",
          "expansion_", expansion_decision_version, "_", variant_file_segment,
          "seed_", seed, "_", tree_label_candidate,
          "_revisit_maxobs_([^_]+)",
          revisit_optional_suffix_regex(),
          "_", input_type, "\\.csv$"
        ),
        paste0(
          "^lambda_([^_]+)_alpha_([^_]+)_beta_([^_]+)_opportunity_([^_]+)_",
          "expansion_", expansion_decision_version, "_", variant_file_segment,
          "seed_", seed, "_", tree_label_candidate,
          "_revisit_maxobs_([^_]+)",
          revisit_optional_suffix_regex(),
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
                  for (critic_suffix in sampled_lambda_critic_file_suffixes()) {
                    suffixes <- if (!is.na(sigma_num) && abs(sigma_num) < 1e-12) {
                      c(critic_suffix, paste0("_obs_sigma_", sigma_candidate, critic_suffix))
                    } else {
                      paste0("_obs_sigma_", sigma_candidate, critic_suffix)
                    }
                    suffixes <- node_coverage_suffix_variants(suffixes)
                    suffixes <- visited_lstm_suffix_variants(stop_paid_suffix_variants(suffixes))
                    for (suffix in suffixes) {
                      file_names <- c(
                        sprintf(
                          "loss_scale_%s_alpha_%s_lambda_%s_opportunity_%s_expansion_%s_%sseed_%d_%s_revisit_maxobs_%s%s_%s.csv",
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
                        ),
                        sprintf(
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
  files <- files[vapply(files, critic_file_matches, logical(1))]
  files <- files[vapply(files, node_coverage_aux_file_matches, logical(1))]
  rows <- list()
  for (tree_label_candidate in simulation_tree_file_labels()) {
    for (variant_file_segment in variant_file_segments) {
      patterns <- c(
        paste0(
          "^loss_scale_([^_]+)_alpha_([^_]+)_lambda_([^_]+)_opportunity_([^_]+)_",
          "expansion_", expansion_decision_version, "_", variant_file_segment,
          "seed_([0-9]+)_", tree_label_candidate,
          "_revisit_maxobs_([^_]+)",
          revisit_optional_suffix_regex(),
          "_", input_type, "\\.csv$"
        ),
        paste0(
          "^lambda_([^_]+)_alpha_([^_]+)_beta_([^_]+)_opportunity_([^_]+)_",
          "expansion_", expansion_decision_version, "_", variant_file_segment,
          "seed_([0-9]+)_", tree_label_candidate,
          "_revisit_maxobs_([^_]+)",
          revisit_optional_suffix_regex(),
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
  }
  if (length(seeded_paths) == 0L) {
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

bayesian_reward_support <- function() {
  if (identical(input_type, "binary")) c(0, 1) else c(-4, -3, -2, -1, 1, 2, 3, 4)
}

bayesian_posterior_probs_from_counts <- function(n, s, sigma_value, reward_support) {
  n <- suppressWarnings(as.numeric(n))
  s <- suppressWarnings(as.numeric(s))
  sigma_num <- suppressWarnings(as.numeric(sigma_value))
  reward_support <- suppressWarnings(as.numeric(reward_support))
  out <- matrix(NA_real_, nrow = length(n), ncol = length(reward_support))
  valid <- is.finite(n) & is.finite(s)
  if (!any(valid)) {
    return(out)
  }
  no_obs <- valid & n <= 0
  if (any(no_obs)) {
    out[no_obs, ] <- 1 / length(reward_support)
  }
  observed <- valid & n > 0
  if (!any(observed)) {
    return(out)
  }
  if (is.na(sigma_num) || sigma_num <= 0) {
    observed_mean <- s[observed] / n[observed]
    nearest_index <- vapply(observed_mean, function(value) {
      which.min(abs(reward_support - value))
    }, integer(1))
    observed_rows <- which(observed)
    for (i in seq_along(observed_rows)) {
      out[observed_rows[[i]], nearest_index[[i]]] <- 1
      out[observed_rows[[i]], -nearest_index[[i]]] <- 0
    }
    return(out)
  }
  rewards <- matrix(reward_support, nrow = sum(observed), ncol = length(reward_support), byrow = TRUE)
  n_obs <- matrix(n[observed], nrow = sum(observed), ncol = length(reward_support))
  s_obs <- matrix(s[observed], nrow = sum(observed), ncol = length(reward_support))
  logp <- -(n_obs * rewards^2 - 2 * s_obs * rewards) / (2 * sigma_num^2)
  logp <- logp - apply(logp, 1L, max, na.rm = TRUE)
  probs <- exp(logp)
  probs <- probs / rowSums(probs)
  out[observed, ] <- probs
  out
}

bayesian_best_node_probability <- function(dat, timestep, sigma_value) {
  reward_support <- bayesian_reward_support()
  n_rows <- nrow(dat)
  if (n_rows == 0) {
    return(numeric())
  }
  previous_timesteps <- if (timestep > 1L) seq_len(timestep - 1L) else integer()
  if (length(previous_timesteps) == 0L) {
    n1 <- n2 <- rep(0, n_rows)
    s1 <- s2 <- rep(0, n_rows)
  } else {
    node_cols <- paste0("expanded_node_t", previous_timesteps)
    reward_cols <- paste0("expanded_reward_t", previous_timesteps)
    node_cols <- node_cols[node_cols %in% names(dat)]
    reward_cols <- reward_cols[reward_cols %in% names(dat)]
    keep_len <- min(length(node_cols), length(reward_cols))
    if (keep_len == 0L) {
      n1 <- n2 <- rep(0, n_rows)
      s1 <- s2 <- rep(0, n_rows)
    } else {
      node_cols <- node_cols[seq_len(keep_len)]
      reward_cols <- reward_cols[seq_len(keep_len)]
      node_mat <- as.matrix(data.frame(lapply(dat[node_cols], function(x) suppressWarnings(as.numeric(x)))))
      reward_mat <- as.matrix(data.frame(lapply(dat[reward_cols], function(x) suppressWarnings(as.numeric(x)))))
      valid_reward <- is.finite(reward_mat)
      node1_mat <- node_mat == 1 & valid_reward
      node2_mat <- node_mat == 2 & valid_reward
      node1_mat[is.na(node1_mat)] <- FALSE
      node2_mat[is.na(node2_mat)] <- FALSE
      n1 <- rowSums(node1_mat)
      n2 <- rowSums(node2_mat)
      s1 <- rowSums(ifelse(node1_mat, reward_mat, 0), na.rm = TRUE)
      s2 <- rowSums(ifelse(node2_mat, reward_mat, 0), na.rm = TRUE)
    }
  }
  probs1 <- bayesian_posterior_probs_from_counts(n1, s1, sigma_value, reward_support)
  probs2 <- bayesian_posterior_probs_from_counts(n2, s2, sigma_value, reward_support)
  greater_matrix <- outer(reward_support, reward_support, ">") * 1
  p_gt <- rowSums((probs1 %*% greater_matrix) * probs2)
  pmin(pmax(p_gt, 0), 1)
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
    prob_1 <- bayesian_best_node_probability(dat, timestep, sigma_value)
    fallback_rows <- !is.finite(prob_1)
    if (any(fallback_rows)) {
      fallback_prob <- terminal_choice_prob_from_means(dat[[mean_1_col]], dat[[mean_2_col]])
      prob_1[fallback_rows] <- fallback_prob[fallback_rows]
    }
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

bin_nearest_integer <- function(x) {
  x <- suppressWarnings(as.numeric(x))
  out <- rep(NA_real_, length(x))
  finite <- is.finite(x)
  out[finite] <- round(x[finite])
  out
}

bin_width_two_away_from_zero <- function(x) {
  x <- suppressWarnings(as.numeric(x))
  out <- rep(NA_real_, length(x))
  finite <- is.finite(x)
  out[finite] <- sign(x[finite]) * ceiling(abs(x[finite]) / 2) * 2
  zero <- finite & abs(x) < 1e-12
  out[zero] <- 0
  out
}

bin_mean_other_for_default_heatmap <- function(x) {
  if (is_bandit3 || is_disjoint3x2) {
    return(bin_nearest_integer(x))
  }
  bin_mean_difference(x)
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

actual_reward_matrix_for_trials <- function(trial_data, actual_lookup = NULL) {
  if (nrow(trial_data) == 0) {
    return(matrix(numeric(), nrow = 0, ncol = task_node_count))
  }
  if (is.null(actual_lookup)) {
    actual_lookup <- build_node_actual_reward_lookup(all_data)
  }
  if (!isTRUE(actual_lookup$available)) {
    return(matrix(NA_real_, nrow = nrow(trial_data), ncol = task_node_count))
  }
  actual_reward_mat <- vapply(seq_len(task_node_count), function(node_id) {
    actual_reward_for_nodes(trial_data, rep(node_id, nrow(trial_data)), actual_lookup)
  }, numeric(nrow(trial_data)))
  if (is.null(dim(actual_reward_mat))) {
    actual_reward_mat <- matrix(actual_reward_mat, nrow = nrow(trial_data), ncol = task_node_count)
  }
  actual_reward_mat
}

path_reward_matrix_from_node_rewards <- function(actual_reward_mat) {
  if (nrow(actual_reward_mat) == 0) {
    return(matrix(numeric(), nrow = 0, ncol = task_path_count))
  }
  path_nodes <- task_path_nodes()
  path_reward_mat <- vapply(path_nodes, function(nodes) {
    rowSums(actual_reward_mat[, nodes, drop = FALSE])
  }, numeric(nrow(actual_reward_mat)))
  if (is.null(dim(path_reward_mat))) {
    path_reward_mat <- matrix(path_reward_mat, nrow = nrow(actual_reward_mat), ncol = task_path_count)
  }
  path_reward_mat
}

add_realized_best_path_normalization <- function(trial_data) {
  trial_data$chosen_path_reward <- suppressWarnings(as.numeric(trial_data$V))
  trial_data$realized_best_path_reward <- NA_real_
  trial_data$reward_norm_denominator <- task_reward_norm
  trial_data$normalized_chosen_path_reward <- trial_data$chosen_path_reward / task_reward_norm
  if (nrow(trial_data) == 0) {
    return(trial_data)
  }
  actual_lookup <- build_node_actual_reward_lookup(all_data)
  if (!isTRUE(actual_lookup$available)) {
    return(trial_data)
  }
  actual_reward_mat <- actual_reward_matrix_for_trials(trial_data, actual_lookup)
  path_reward_mat <- path_reward_matrix_from_node_rewards(actual_reward_mat)
  finite_path_rows <- rowSums(is.finite(path_reward_mat)) == ncol(path_reward_mat)
  if (!any(finite_path_rows)) {
    return(trial_data)
  }
  trial_data$realized_best_path_reward[finite_path_rows] <- apply(
    path_reward_mat[finite_path_rows, , drop = FALSE],
    1,
    max
  )
  group_cols <- intersect(c("model", "beta", "opportunity", "sigma", "seed"), names(trial_data))
  if (length(group_cols) == 0) {
    group_key <- rep("all", nrow(trial_data))
  } else {
    group_key <- do.call(paste, c(lapply(group_cols, function(col) as.character(trial_data[[col]])), sep = "\r"))
  }
  group_levels <- unique(group_key)
  for (key in group_levels) {
    rows <- which(group_key == key)
    denom <- mean(trial_data$realized_best_path_reward[rows], na.rm = TRUE)
    if (is.finite(denom) && abs(denom) > 1e-12) {
      trial_data$reward_norm_denominator[rows] <- denom
    }
  }
  valid_denom <- is.finite(trial_data$reward_norm_denominator) &
    abs(trial_data$reward_norm_denominator) > 1e-12
  trial_data$normalized_chosen_path_reward[valid_denom] <-
    trial_data$chosen_path_reward[valid_denom] / trial_data$reward_norm_denominator[valid_denom]
  trial_data
}

actual_path_reward_tie_flags <- function(trial_data, tol = 1e-8) {
  out <- rep(FALSE, nrow(trial_data))
  if (nrow(trial_data) == 0 || task_path_count < 2L) {
    return(out)
  }
  actual_lookup <- build_node_actual_reward_lookup(all_data)
  if (!isTRUE(actual_lookup$available)) {
    warning("Missing node/actual_reward columns; tied actual path rewards cannot be removed from entropy plots.")
    return(out)
  }
  actual_reward_mat <- actual_reward_matrix_for_trials(trial_data, actual_lookup)
  path_reward_mat <- path_reward_matrix_from_node_rewards(actual_reward_mat)
  finite_rows <- rowSums(is.finite(path_reward_mat)) == ncol(path_reward_mat)
  if (!any(finite_rows)) {
    return(out)
  }
  if (ncol(path_reward_mat) == 2L) {
    out[finite_rows] <- abs(path_reward_mat[finite_rows, 1L] - path_reward_mat[finite_rows, 2L]) <= tol
    return(out)
  }
  path_pairs <- utils::combn(seq_len(ncol(path_reward_mat)), 2L)
  pairwise_ties <- abs(path_reward_mat[, path_pairs[1L, ], drop = FALSE] -
    path_reward_mat[, path_pairs[2L, ], drop = FALSE]) <= tol
  out[finite_rows] <- rowSums(pairwise_ties[finite_rows, , drop = FALSE], na.rm = TRUE) > 0
  out
}

first_observed_path_context_values <- function(trial_data, node_mat) {
  empty <- list(
    first_reward = rep(NA_real_, nrow(trial_data)),
    other_mean = rep(NA_real_, nrow(trial_data)),
    signed = rep(NA_real_, nrow(trial_data)),
    absolute = rep(NA_real_, nrow(trial_data))
  )
  if (nrow(trial_data) == 0 || ncol(node_mat) == 0 || task_path_count < 2L) {
    return(empty)
  }
  actual_lookup <- build_node_actual_reward_lookup(all_data)
  if (!isTRUE(actual_lookup$available)) {
    return(empty)
  }
  actual_reward_mat <- actual_reward_matrix_for_trials(trial_data, actual_lookup)
  path_reward_mat <- path_reward_matrix_from_node_rewards(actual_reward_mat)
  path_mat <- matrix(
    path_id_for_node(as.vector(node_mat)),
    nrow = nrow(node_mat),
    ncol = ncol(node_mat)
  )
  finite_path_mat <- is.finite(path_mat)
  if (!any(finite_path_mat)) {
    return(empty)
  }
  path_col_indices <- matrix(seq_len(ncol(path_mat)),
    nrow = nrow(path_mat),
    ncol = ncol(path_mat),
    byrow = TRUE
  )
  first_path_col <- max.col(ifelse(finite_path_mat, ncol(path_mat) - path_col_indices + 1L, 0),
    ties.method = "first"
  )
  has_path <- rowSums(finite_path_mat) > 0
  first_path <- rep(NA_integer_, nrow(path_mat))
  first_path[has_path] <- suppressWarnings(as.integer(path_mat[cbind(which(has_path), first_path_col[has_path])]))
  valid_first_path <- !is.na(first_path) & first_path >= 1L & first_path <= ncol(path_reward_mat)
  if (!any(valid_first_path)) {
    return(empty)
  }
  rows <- which(valid_first_path)
  first_reward <- rep(NA_real_, nrow(trial_data))
  first_reward[rows] <- path_reward_mat[cbind(rows, first_path[rows])]
  path_reward_sums <- rowSums(path_reward_mat)
  other_mean <- rep(NA_real_, nrow(trial_data))
  other_mean[rows] <- (path_reward_sums[rows] - first_reward[rows]) / (task_path_count - 1L)
  signed <- first_reward - other_mean
  list(
    first_reward = first_reward,
    other_mean = other_mean,
    signed = signed,
    absolute = abs(signed)
  )
}

first_observed_path_difference_values <- function(trial_data, node_mat) {
  context <- first_observed_path_context_values(trial_data, node_mat)
  list(
    signed = context$signed,
    absolute = context$absolute
  )
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
trial_data <- add_realized_best_path_normalization(trial_data)
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
trial_data$actual_path_reward_tied <- actual_path_reward_tie_flags(trial_data)
num_tied_actual_path_reward_trials <- sum(as_logical_col(trial_data$actual_path_reward_tied), na.rm = TRUE)
if (num_tied_actual_path_reward_trials > 0L) {
  message(sprintf(
    "Excluding %d trial(s) with tied actual path rewards from terminal-choice entropy/probability plots.",
    num_tied_actual_path_reward_trials
  ))
}

max_observations_before_stop_num <- suppressWarnings(as.numeric(max_observations_arg))
max_entropy_observation_timestep <- if (is.finite(max_observations_before_stop_num)) {
  max_observations_before_stop_num
} else {
  Inf
}
max_entropy_terminal_timestep <- max_entropy_observation_timestep
cap_terminal_entropy_timestep <- function(timestep) {
  timestep <- suppressWarnings(as.integer(timestep))
  if (is.finite(max_entropy_terminal_timestep)) {
    return(as.integer(pmin(timestep, max_entropy_terminal_timestep)))
  }
  timestep
}
include_entropy_observation_timestep <- function(timestep) {
  timestep <- suppressWarnings(as.numeric(timestep))
  is.finite(timestep) & timestep > 0 & timestep < max_entropy_observation_timestep
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
  tied_rows <- "actual_path_reward_tied" %in% names(trial_data) &
    as_logical_col(trial_data$actual_path_reward_tied)
  entropy[tied_rows] <- NA_real_
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
  tied_rows <- "actual_path_reward_tied" %in% names(trial_data) &
    as_logical_col(trial_data$actual_path_reward_tied)
  entropy[tied_rows] <- NA_real_
  entropy
}

terminal_binary_choice_prob_extrema_for_timestep <- function(timestep) {
  prob_cols <- terminal_prob_cols_for_timestep(timestep)
  empty <- list(
    max_prob = rep(NA_real_, nrow(trial_data)),
    min_prob = rep(NA_real_, nrow(trial_data))
  )
  if (length(prob_cols) < 2L) {
    return(empty)
  }
  prob_cols <- prob_cols[seq_len(2L)]
  prob_mat <- numeric_matrix_from_cols(trial_data, prob_cols)
  prob_mat[!is.finite(prob_mat) | prob_mat <= 0] <- 0
  prob_sums <- rowSums(prob_mat)
  valid_rows <- is.finite(prob_sums) & prob_sums > 0
  max_prob <- rep(NA_real_, nrow(prob_mat))
  min_prob <- rep(NA_real_, nrow(prob_mat))
  if (any(valid_rows)) {
    normalized_prob_mat <- prob_mat[valid_rows, , drop = FALSE] / prob_sums[valid_rows]
    max_prob[valid_rows] <- pmax(normalized_prob_mat[, 1L], normalized_prob_mat[, 2L])
    min_prob[valid_rows] <- pmin(normalized_prob_mat[, 1L], normalized_prob_mat[, 2L])
  }
  tied_rows <- "actual_path_reward_tied" %in% names(trial_data) &
    as_logical_col(trial_data$actual_path_reward_tied)
  max_prob[tied_rows] <- NA_real_
  min_prob[tied_rows] <- NA_real_
  list(max_prob = max_prob, min_prob = min_prob)
}

terminal_prob_timesteps <- terminal_prob_timesteps[is.finite(terminal_prob_timesteps)]
if (length(terminal_prob_timesteps) > 0) {
  for (timestep in terminal_prob_timesteps) {
    stop_col <- paste0("stop_t", timestep)
    if (ncol(stop_mat) == 0 || !stop_col %in% names(trial_data)) {
      next
    }
    entropy_t <- terminal_choice_entropy_for_timestep(cap_terminal_entropy_timestep(timestep))
    stopped_t <- as_logical_col(trial_data[[stop_col]])
    fill_rows <- stopped_t & !is.finite(trial_data$terminal_choice_entropy)
    trial_data$terminal_choice_entropy[fill_rows] <- entropy_t[fill_rows]
  }
  last_terminal_timestep <- max(terminal_prob_timesteps, na.rm = TRUE)
  last_entropy <- terminal_choice_entropy_for_timestep(cap_terminal_entropy_timestep(last_terminal_timestep))
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
first_observed_path_context <- first_observed_path_context_values(trial_data, node_mat)
trial_data$first_observed_path_actual_reward_raw <- first_observed_path_context$first_reward
trial_data$mean_other_path_actual_reward_raw <- first_observed_path_context$other_mean
trial_data$first_observed_path_actual_reward <- bin_mean_difference(first_observed_path_context$first_reward)
trial_data$mean_other_path_actual_reward <- bin_mean_other_for_default_heatmap(first_observed_path_context$other_mean)
trial_data$first_observed_path_actual_reward_integer <- bin_nearest_integer(first_observed_path_context$first_reward)
trial_data$mean_other_path_actual_reward_integer <- bin_nearest_integer(first_observed_path_context$other_mean)
trial_data$first_observed_path_actual_reward_bin2 <- bin_width_two_away_from_zero(first_observed_path_context$first_reward)
trial_data$mean_other_path_actual_reward_bin2 <- bin_width_two_away_from_zero(first_observed_path_context$other_mean)
trial_data$first_observed_minus_mean_other_path <- bin_mean_difference(first_observed_path_context$signed)
trial_data$absolute_first_observed_minus_mean_other_path <- bin_mean_difference(first_observed_path_context$absolute)

mean_or_na <- function(x) {
  x <- suppressWarnings(as.numeric(x))
  x <- x[is.finite(x)]
  if (length(x) == 0) NA_real_ else mean(x)
}

sem_or_na <- function(x) {
  x <- suppressWarnings(as.numeric(x))
  x <- x[is.finite(x)]
  if (length(x) < 2L) NA_real_ else stats::sd(x) / sqrt(length(x))
}

log10_sem_or_na <- function(x) {
  x <- suppressWarnings(as.numeric(x))
  x <- x[is.finite(x) & x > 0]
  if (length(x) < 2L) {
    return(NA_real_)
  }
  stats::sd(log10(pmax(x, log_axis_floor))) / sqrt(length(x))
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
    # terminal_choice_prob_path*_t is post-observation for observe actions,
    # while kl_d_t+1 is the carried-forward KL paid if the trial continues.
    decision_timestep <- observation_timestep
    kl_col <- paste0("kl_d_t", observation_timestep + 1L)
    kl_values <- if (kl_col %in% names(trial_data)) {
      suppressWarnings(as.numeric(trial_data[[kl_col]]))
    } else {
      rep(NA_real_, nrow(trial_data))
    }
    kl_values <- ifelse(
      suppressWarnings(as.numeric(trial_data$timestep_before_stop)) > observation_timestep,
      kl_values,
      NA_real_
    )
    entropy_values <- terminal_binary_choice_entropy_for_timestep(decision_timestep)
    full_entropy_values <- terminal_choice_entropy_for_timestep(decision_timestep)
    prob_extrema <- terminal_binary_choice_prob_extrema_for_timestep(decision_timestep)
    rows[[length(rows) + 1L]] <- data.frame(
      model = trial_data$model[before_stop],
      sigma = trial_data$sigma[before_stop],
      beta = trial_data$beta[before_stop],
      opportunity = trial_data$opportunity[before_stop],
      seed = trial_data$seed[before_stop],
      timestep = observation_timestep,
      decision_timestep = decision_timestep,
      timestep_before_stop = trial_data$timestep_before_stop[before_stop],
      first_observed_path_actual_reward = trial_data$first_observed_path_actual_reward[before_stop],
      mean_other_path_actual_reward = trial_data$mean_other_path_actual_reward[before_stop],
      first_observed_path_actual_reward_raw = trial_data$first_observed_path_actual_reward_raw[before_stop],
      mean_other_path_actual_reward_raw = trial_data$mean_other_path_actual_reward_raw[before_stop],
      first_observed_path_actual_reward_integer = trial_data$first_observed_path_actual_reward_integer[before_stop],
      mean_other_path_actual_reward_integer = trial_data$mean_other_path_actual_reward_integer[before_stop],
      first_observed_path_actual_reward_bin2 = trial_data$first_observed_path_actual_reward_bin2[before_stop],
      mean_other_path_actual_reward_bin2 = trial_data$mean_other_path_actual_reward_bin2[before_stop],
      absolute_first_observed_minus_mean_other_path = trial_data$absolute_first_observed_minus_mean_other_path[before_stop],
      actual_path_reward_tied = trial_data$actual_path_reward_tied[before_stop],
      kl_paid_at_timestep = kl_values[before_stop],
      terminal_choice_entropy_at_timestep = full_entropy_values[before_stop],
      terminal_binary_choice_entropy_at_timestep = entropy_values[before_stop],
      terminal_choice_prob_max_at_timestep = prob_extrema$max_prob[before_stop],
      terminal_choice_prob_min_at_timestep = prob_extrema$min_prob[before_stop],
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

add_seed_sem_by <- function(summary_data, dat, group_cols, value_cols) {
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
    sd_data <- dt[, lapply(.SD, sem_or_na), by = group_cols, .SDcols = value_cols]
    sd_data <- as.data.frame(sd_data)
    log_sem_data <- dt[, lapply(.SD, log10_sem_or_na), by = group_cols, .SDcols = value_cols]
    log_sem_data <- as.data.frame(log_sem_data)
  } else {
    pieces <- lapply(value_cols, function(value_col) {
      out <- aggregate(seed_means[[value_col]], by = seed_means[, group_cols, drop = FALSE], FUN = sem_or_na)
      names(out)[names(out) == "x"] <- value_col
      out
    })
    sd_data <- Reduce(function(left, right) merge(left, right, by = group_cols, all = TRUE), pieces)
    log_pieces <- lapply(value_cols, function(value_col) {
      out <- aggregate(seed_means[[value_col]], by = seed_means[, group_cols, drop = FALSE], FUN = log10_sem_or_na)
      names(out)[names(out) == "x"] <- value_col
      out
    })
    log_sem_data <- Reduce(function(left, right) merge(left, right, by = group_cols, all = TRUE), log_pieces)
  }
  names(sd_data)[names(sd_data) %in% value_cols] <- paste0(names(sd_data)[names(sd_data) %in% value_cols], "_seed_sem")
  names(log_sem_data)[names(log_sem_data) %in% value_cols] <- paste0(
    names(log_sem_data)[names(log_sem_data) %in% value_cols],
    "_seed_log10_sem"
  )
  summary_data <- merge(summary_data, sd_data, by = group_cols, all.x = TRUE)
  merge(summary_data, log_sem_data, by = group_cols, all.x = TRUE)
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
  pre_stop_timestep_value_cols <- c(
    "kl_paid_at_timestep",
    "terminal_binary_choice_entropy_at_timestep",
    "terminal_choice_prob_max_at_timestep",
    "terminal_choice_prob_min_at_timestep"
  )
  pre_stop_timestep_summary <- aggregate_means_by(
    pre_stop_timestep_data,
    group_cols = c("model", "sigma", "beta", "opportunity", "timestep"),
    value_cols = pre_stop_timestep_value_cols
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
  pre_stop_timestep_summary <- add_seed_sem_by(
    pre_stop_timestep_summary,
    pre_stop_timestep_data,
    group_cols = c("model", "sigma", "beta", "opportunity", "timestep"),
    value_cols = pre_stop_timestep_value_cols
  )
} else {
  pre_stop_timestep_summary <- data.frame()
}
if (nrow(pre_stop_timestep_data) > 0) {
  pre_stop_timestep_by_total_data <- pre_stop_timestep_data[
    is.finite(suppressWarnings(as.numeric(pre_stop_timestep_data$timestep_before_stop))) &
      suppressWarnings(as.numeric(pre_stop_timestep_data$timestep_before_stop)) > 0,
    ,
    drop = FALSE
  ]
} else {
  pre_stop_timestep_by_total_data <- data.frame()
}
if (nrow(pre_stop_timestep_by_total_data) > 0) {
  pre_stop_timestep_by_total_summary <- aggregate_means_by(
    pre_stop_timestep_by_total_data,
    group_cols = c("model", "sigma", "beta", "opportunity", "timestep_before_stop", "timestep"),
    value_cols = c("kl_paid_at_timestep")
  )
  pre_stop_timestep_by_total_count <- count_rows_by(
    pre_stop_timestep_by_total_data,
    c("model", "sigma", "beta", "opportunity", "timestep_before_stop", "timestep")
  )
  pre_stop_timestep_by_total_summary <- merge(
    pre_stop_timestep_by_total_summary,
    pre_stop_timestep_by_total_count,
    by = c("model", "sigma", "beta", "opportunity", "timestep_before_stop", "timestep"),
    all.x = TRUE
  )
  pre_stop_timestep_by_total_summary <- add_seed_sem_by(
    pre_stop_timestep_by_total_summary,
    pre_stop_timestep_by_total_data,
    group_cols = c("model", "sigma", "beta", "opportunity", "timestep_before_stop", "timestep"),
    value_cols = c("kl_paid_at_timestep")
  )
} else {
  pre_stop_timestep_by_total_summary <- data.frame()
}
pre_stop_timestep_entropy_data <- if (nrow(pre_stop_timestep_data) > 0 && "actual_path_reward_tied" %in% names(pre_stop_timestep_data)) {
  pre_stop_timestep_data[!as_logical_col(pre_stop_timestep_data$actual_path_reward_tied), , drop = FALSE]
} else {
  pre_stop_timestep_data
}
if (nrow(pre_stop_timestep_entropy_data) > 0 && "timestep" %in% names(pre_stop_timestep_entropy_data)) {
  pre_stop_timestep_entropy_data <- pre_stop_timestep_entropy_data[
    include_entropy_observation_timestep(pre_stop_timestep_entropy_data$timestep),
    ,
    drop = FALSE
  ]
}
pre_stop_terminal_combo_value_cols <- c(
  "terminal_binary_choice_entropy_at_timestep",
  "terminal_choice_prob_max_at_timestep",
  "terminal_choice_prob_min_at_timestep"
)
if (nrow(pre_stop_timestep_entropy_data) > 0) {
  pre_stop_timestep_entropy_summary <- aggregate_means_by(
    pre_stop_timestep_entropy_data,
    group_cols = c("model", "sigma", "beta", "opportunity", "timestep"),
    value_cols = pre_stop_terminal_combo_value_cols
  )
  pre_stop_timestep_entropy_count <- count_rows_by(
    pre_stop_timestep_entropy_data,
    c("model", "sigma", "beta", "opportunity", "timestep")
  )
  pre_stop_timestep_entropy_summary <- merge(
    pre_stop_timestep_entropy_summary,
    pre_stop_timestep_entropy_count,
    by = c("model", "sigma", "beta", "opportunity", "timestep"),
    all.x = TRUE
  )
  pre_stop_timestep_entropy_summary <- add_seed_sem_by(
    pre_stop_timestep_entropy_summary,
    pre_stop_timestep_entropy_data,
    group_cols = c("model", "sigma", "beta", "opportunity", "timestep"),
    value_cols = pre_stop_terminal_combo_value_cols
  )
  pre_stop_entropy_combo_summary <- aggregate_means_by(
    pre_stop_timestep_entropy_data,
    group_cols = c("model", "sigma", "beta", "opportunity"),
    value_cols = pre_stop_terminal_combo_value_cols
  )
  pre_stop_entropy_combo_count <- count_rows_by(
    pre_stop_timestep_entropy_data,
    c("model", "sigma", "beta", "opportunity")
  )
  pre_stop_entropy_combo_summary <- merge(
    pre_stop_entropy_combo_summary,
    pre_stop_entropy_combo_count,
    by = c("model", "sigma", "beta", "opportunity"),
    all.x = TRUE
  )
  names(pre_stop_entropy_combo_summary)[
    names(pre_stop_entropy_combo_summary) == "terminal_binary_choice_entropy_at_timestep"
  ] <- "terminal_choice_entropy_combined_reached"
  names(pre_stop_entropy_combo_summary)[
    names(pre_stop_entropy_combo_summary) == "terminal_choice_prob_max_at_timestep"
  ] <- "terminal_choice_prob_max_combined_reached"
  names(pre_stop_entropy_combo_summary)[
    names(pre_stop_entropy_combo_summary) == "terminal_choice_prob_min_at_timestep"
  ] <- "terminal_choice_prob_min_combined_reached"
  pre_stop_entropy_combo_summary <- add_seed_sem_by(
    pre_stop_entropy_combo_summary,
    pre_stop_timestep_entropy_data,
    group_cols = c("model", "sigma", "beta", "opportunity"),
    value_cols = pre_stop_terminal_combo_value_cols
  )
  names(pre_stop_entropy_combo_summary)[
    names(pre_stop_entropy_combo_summary) == "terminal_binary_choice_entropy_at_timestep_seed_sem"
  ] <- "terminal_choice_entropy_combined_reached_seed_sem"
  names(pre_stop_entropy_combo_summary)[
    names(pre_stop_entropy_combo_summary) == "terminal_choice_prob_max_at_timestep_seed_sem"
  ] <- "terminal_choice_prob_max_combined_reached_seed_sem"
  names(pre_stop_entropy_combo_summary)[
    names(pre_stop_entropy_combo_summary) == "terminal_choice_prob_min_at_timestep_seed_sem"
  ] <- "terminal_choice_prob_min_combined_reached_seed_sem"
} else {
  pre_stop_timestep_entropy_summary <- data.frame()
  pre_stop_entropy_combo_summary <- data.frame()
}
pre_stop_timestep_entropy_strict_data <- if (
  nrow(pre_stop_timestep_entropy_data) > 0 &&
    all(c("timestep", "timestep_before_stop") %in% names(pre_stop_timestep_entropy_data))
) {
  pre_stop_timestep_entropy_data[
    suppressWarnings(as.numeric(pre_stop_timestep_entropy_data$timestep)) <
      suppressWarnings(as.numeric(pre_stop_timestep_entropy_data$timestep_before_stop)),
    ,
    drop = FALSE
  ]
} else {
  data.frame()
}
if (nrow(pre_stop_timestep_entropy_strict_data) > 0) {
  pre_stop_timestep_entropy_strict_summary <- aggregate_means_by(
    pre_stop_timestep_entropy_strict_data,
    group_cols = c("model", "sigma", "beta", "opportunity", "timestep"),
    value_cols = pre_stop_terminal_combo_value_cols
  )
  pre_stop_timestep_entropy_strict_count <- count_rows_by(
    pre_stop_timestep_entropy_strict_data,
    c("model", "sigma", "beta", "opportunity", "timestep")
  )
  pre_stop_timestep_entropy_strict_summary <- merge(
    pre_stop_timestep_entropy_strict_summary,
    pre_stop_timestep_entropy_strict_count,
    by = c("model", "sigma", "beta", "opportunity", "timestep"),
    all.x = TRUE
  )
  pre_stop_timestep_entropy_strict_summary <- add_seed_sem_by(
    pre_stop_timestep_entropy_strict_summary,
    pre_stop_timestep_entropy_strict_data,
    group_cols = c("model", "sigma", "beta", "opportunity", "timestep"),
    value_cols = pre_stop_terminal_combo_value_cols
  )
} else {
  pre_stop_timestep_entropy_strict_summary <- data.frame()
}
pre_stop_timestep_entropy_strict_by_absdiff_data <- if (
  nrow(pre_stop_timestep_entropy_strict_data) > 0 &&
    "absolute_first_observed_minus_mean_other_path" %in% names(pre_stop_timestep_entropy_strict_data)
) {
  pre_stop_timestep_entropy_strict_data[
    is.finite(suppressWarnings(as.numeric(pre_stop_timestep_entropy_strict_data$absolute_first_observed_minus_mean_other_path))),
    ,
    drop = FALSE
  ]
} else {
  data.frame()
}
if (nrow(pre_stop_timestep_entropy_strict_by_absdiff_data) > 0) {
  pre_stop_timestep_strict_by_absdiff_value_cols <- unique(c(
    "kl_paid_at_timestep",
    pre_stop_terminal_combo_value_cols
  ))
  pre_stop_timestep_entropy_strict_by_absdiff_summary <- aggregate_means_by(
    pre_stop_timestep_entropy_strict_by_absdiff_data,
    group_cols = c(
      "model", "sigma", "beta", "opportunity", "timestep",
      "absolute_first_observed_minus_mean_other_path"
    ),
    value_cols = pre_stop_timestep_strict_by_absdiff_value_cols
  )
  pre_stop_timestep_entropy_strict_by_absdiff_count <- count_rows_by(
    pre_stop_timestep_entropy_strict_by_absdiff_data,
    c(
      "model", "sigma", "beta", "opportunity", "timestep",
      "absolute_first_observed_minus_mean_other_path"
    )
  )
  pre_stop_timestep_entropy_strict_by_absdiff_summary <- merge(
    pre_stop_timestep_entropy_strict_by_absdiff_summary,
    pre_stop_timestep_entropy_strict_by_absdiff_count,
    by = c(
      "model", "sigma", "beta", "opportunity", "timestep",
      "absolute_first_observed_minus_mean_other_path"
    ),
    all.x = TRUE
  )
  pre_stop_timestep_entropy_strict_by_absdiff_summary <- add_seed_sem_by(
    pre_stop_timestep_entropy_strict_by_absdiff_summary,
    pre_stop_timestep_entropy_strict_by_absdiff_data,
    group_cols = c(
      "model", "sigma", "beta", "opportunity", "timestep",
      "absolute_first_observed_minus_mean_other_path"
    ),
    value_cols = pre_stop_timestep_strict_by_absdiff_value_cols
  )
} else {
  pre_stop_timestep_entropy_strict_by_absdiff_summary <- data.frame()
}

summarize_heatmap_cells <- function(
  dat,
  value_col,
  include_timestep = TRUE,
  y_coord_col = "first_observed_path_actual_reward",
  x_coord_col = "mean_other_path_actual_reward"
) {
  coord_cols <- c("first_observed_path_actual_reward", "mean_other_path_actual_reward")
  required_cols <- unique(c(
    "model", "sigma", "beta", "opportunity", y_coord_col, x_coord_col, value_col,
    if (isTRUE(include_timestep)) "timestep" else character()
  ))
  if (nrow(dat) == 0 || !all(required_cols %in% names(dat))) {
    return(data.frame())
  }
  work <- dat
  work$first_observed_path_actual_reward <- suppressWarnings(as.numeric(work[[y_coord_col]]))
  work$mean_other_path_actual_reward <- suppressWarnings(as.numeric(work[[x_coord_col]]))
  for (coord_col in coord_cols) {
    work[[coord_col]] <- suppressWarnings(as.numeric(work[[coord_col]]))
  }
  work[[value_col]] <- suppressWarnings(as.numeric(work[[value_col]]))
  keep <- is.finite(work[[value_col]]) &
    is.finite(work$first_observed_path_actual_reward) &
    is.finite(work$mean_other_path_actual_reward)
  if (isTRUE(include_timestep)) {
    work$timestep <- suppressWarnings(as.numeric(work$timestep))
    keep <- keep & is.finite(work$timestep)
  }
  work <- work[keep, , drop = FALSE]
  if (nrow(work) == 0) {
    return(data.frame())
  }
  group_cols <- c("model", "sigma", "beta", "opportunity")
  if (isTRUE(include_timestep)) {
    group_cols <- c(group_cols, "timestep")
  }
  group_cols <- c(group_cols, coord_cols)
  out <- aggregate_means_by(
    work,
    group_cols = group_cols,
    value_cols = value_col
  )
  value_count <- count_rows_by(
    work,
    group_cols,
    count_name = paste0(value_col, "_n")
  )
  merge(out, value_count, by = group_cols, all.x = TRUE)
}

build_heatmap_summary_set <- function(
  y_coord_col = "first_observed_path_actual_reward",
  x_coord_col = "mean_other_path_actual_reward"
) {
  list(
    terminal_entropy = summarize_heatmap_cells(
      pre_stop_timestep_entropy_strict_data,
      "terminal_choice_entropy_at_timestep",
      include_timestep = TRUE,
      y_coord_col = y_coord_col,
      x_coord_col = x_coord_col
    ),
    kl_timestep = summarize_heatmap_cells(
      pre_stop_timestep_data,
      "kl_paid_at_timestep",
      include_timestep = TRUE,
      y_coord_col = y_coord_col,
      x_coord_col = x_coord_col
    ),
    timestep_before_stop = summarize_heatmap_cells(
      trial_data,
      "timestep_before_stop",
      include_timestep = FALSE,
      y_coord_col = y_coord_col,
      x_coord_col = x_coord_col
    )
  )
}

if (isTRUE(selected_revisit_plots_only)) {
  default_heatmap_summaries <- list(
    terminal_entropy = data.frame(),
    kl_timestep = data.frame(),
    timestep_before_stop = data.frame()
  )
  disjoint_integer_heatmap_summaries <- NULL
  disjoint_bin2_heatmap_summaries <- NULL
} else {
  default_heatmap_summaries <- build_heatmap_summary_set()
  disjoint_integer_heatmap_summaries <- if (is_disjoint2x2 || is_disjoint3x2) {
    build_heatmap_summary_set(
      y_coord_col = "first_observed_path_actual_reward_raw",
      x_coord_col = "mean_other_path_actual_reward_integer"
    )
  } else {
    NULL
  }
  disjoint_bin2_heatmap_summaries <- if (is_disjoint2x2 || is_disjoint3x2) {
    build_heatmap_summary_set(
      y_coord_col = "first_observed_path_actual_reward_bin2",
      x_coord_col = "mean_other_path_actual_reward_bin2"
    )
  } else {
    NULL
  }
}
terminal_entropy_heatmap_summary <- default_heatmap_summaries$terminal_entropy
kl_timestep_heatmap_summary <- default_heatmap_summaries$kl_timestep
timestep_before_stop_heatmap_summary <- default_heatmap_summaries$timestep_before_stop

build_first_timestep_kl_after_continue_summary <- function() {
  required_cols <- c("expanded_node_t1", "expanded_reward_t1", "kl_d_t2", "timestep_before_stop")
  if (!all(required_cols %in% names(trial_data))) {
    return(data.frame())
  }
  continued_after_first_reward <- is.finite(suppressWarnings(as.numeric(trial_data$timestep_before_stop))) &
    suppressWarnings(as.numeric(trial_data$timestep_before_stop)) > 1
  first_reward <- observed_actual_reward_values(
    trial_data,
    node_col = "expanded_node_t1",
    reward_col = "expanded_reward_t1",
    actual_lookup = build_node_actual_reward_lookup(all_data)
  )
  kl_first <- suppressWarnings(as.numeric(trial_data$kl_d_t2))
  keep <- continued_after_first_reward & is.finite(first_reward) & is.finite(kl_first)
  if (!any(keep)) {
    return(data.frame())
  }
  first_kl_data <- data.frame(
    model = trial_data$model[keep],
    sigma = trial_data$sigma[keep],
    beta = trial_data$beta[keep],
    opportunity = trial_data$opportunity[keep],
    seed = trial_data$seed[keep],
    first_observed_reward_t1 = bin_mean_difference(first_reward[keep]),
    kl_paid_at_first_timestep_after_continue = kl_first[keep],
    stringsAsFactors = FALSE
  )
  group_cols <- c("model", "sigma", "beta", "opportunity", "first_observed_reward_t1")
  value_cols <- c("kl_paid_at_first_timestep_after_continue")
  out <- aggregate_means_by(
    first_kl_data,
    group_cols = group_cols,
    value_cols = value_cols
  )
  count_data <- count_rows_by(first_kl_data, group_cols)
  out <- merge(out, count_data, by = group_cols, all.x = TRUE)
  value_count_data <- count_rows_by(
    first_kl_data[is.finite(first_kl_data$kl_paid_at_first_timestep_after_continue), , drop = FALSE],
    group_cols,
    count_name = "kl_paid_at_first_timestep_after_continue_n"
  )
  out <- merge(out, value_count_data, by = group_cols, all.x = TRUE)
  add_seed_sem_by(
    out,
    first_kl_data,
    group_cols = group_cols,
    value_cols = value_cols
  )
}

first_timestep_kl_after_continue_summary <- build_first_timestep_kl_after_continue_summary()

average_value_cols <- c(
  "normalized_chosen_path_reward",
  "observations_before_stop",
  "stop_decision_timestep",
  "timestep_before_stop",
  "unique_nodes_visited",
  "kl_paid_total",
  "terminal_choice_entropy"
)
average_summary <- aggregate_means_by(
  trial_data,
  group_cols = c("model", "sigma", "beta", "opportunity"),
  value_cols = average_value_cols
)
count_data <- count_rows_by(trial_data, c("model", "sigma", "beta", "opportunity"))
average_summary <- merge(average_summary, count_data, by = c("model", "sigma", "beta", "opportunity"), all.x = TRUE)
average_summary <- add_seed_sem_by(
  average_summary,
  trial_data,
  group_cols = c("model", "sigma", "beta", "opportunity"),
  value_cols = average_value_cols
)
if (nrow(pre_stop_entropy_combo_summary) > 0) {
  average_summary <- merge(
    average_summary,
    pre_stop_entropy_combo_summary,
    by = c("model", "sigma", "beta", "opportunity"),
    all.x = TRUE
  )
}

entropy_trial_data <- if ("actual_path_reward_tied" %in% names(trial_data)) {
  trial_data[!as_logical_col(trial_data$actual_path_reward_tied), , drop = FALSE]
} else {
  trial_data
}
entropy_average_value_cols <- c(
  "kl_paid_total",
  "timestep_before_stop",
  "terminal_choice_entropy"
)
if (nrow(entropy_trial_data) > 0) {
  entropy_average_summary <- aggregate_means_by(
    entropy_trial_data,
    group_cols = c("model", "sigma", "beta", "opportunity"),
    value_cols = entropy_average_value_cols
  )
  entropy_count_data <- count_rows_by(entropy_trial_data, c("model", "sigma", "beta", "opportunity"))
  entropy_average_summary <- merge(
    entropy_average_summary,
    entropy_count_data,
    by = c("model", "sigma", "beta", "opportunity"),
    all.x = TRUE
  )
  entropy_average_summary <- add_seed_sem_by(
    entropy_average_summary,
    entropy_trial_data,
    group_cols = c("model", "sigma", "beta", "opportunity"),
    value_cols = entropy_average_value_cols
  )
  if (nrow(pre_stop_entropy_combo_summary) > 0) {
    entropy_average_summary <- merge(
      entropy_average_summary,
      pre_stop_entropy_combo_summary,
      by = c("model", "sigma", "beta", "opportunity"),
      all.x = TRUE,
      suffixes = c("", "_pre_stop")
    )
    if ("n_pre_stop" %in% names(entropy_average_summary)) {
      entropy_average_summary$n_pre_stop <- NULL
    }
  }
} else {
  entropy_average_summary <- data.frame()
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

is_timestep_axis_col <- function(x_col) {
  x_col %in% c("timestep", "strict_pre_stop_timestep", "timestep_before_stop")
}

timestep_x_limits <- function(x) {
  x <- suppressWarnings(as.numeric(x))
  x <- x[is.finite(x)]
  if (length(x) == 0L) {
    return(c(1, 2))
  }
  max_tick <- max(1L, ceiling(max(x)))
  c(1, if (max_tick <= 1L) 2 else max_tick)
}

timestep_x_ticks <- function(x) {
  x <- suppressWarnings(as.numeric(x))
  x <- x[is.finite(x)]
  max_tick <- if (length(x) == 0L) 1L else max(1L, ceiling(max(x)))
  seq.int(1L, max_tick)
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
    cex = 1
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
    cex = 1
  )
}

log_arg_for_axes <- function(log_x = FALSE, log_y = FALSE) {
  paste0(if (isTRUE(log_x)) "x" else "", if (isTRUE(log_y)) "y" else "")
}

passes_min_sample_threshold <- function(dat, n_col) {
  if (is.null(dat) || nrow(dat) == 0 || is.na(n_col) || !n_col %in% names(dat)) {
    return(rep(TRUE, if (is.null(dat)) 0L else nrow(dat)))
  }
  sample_n <- suppressWarnings(as.numeric(dat[[n_col]]))
  optimal_rows <- if ("model" %in% names(dat)) {
    as.character(dat$model) == "Optimal"
  } else {
    rep(FALSE, nrow(dat))
  }
  optimal_rows | (is.finite(sample_n) & sample_n >= minimum_samples_threshold)
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
  metric_n_col <- paste0(y_col, "_n")
  n_col <- if (metric_n_col %in% names(dat)) {
    metric_n_col
  } else if ("n" %in% names(dat)) {
    "n"
  } else {
    NA_character_
  }
  if (!is.na(n_col)) {
    keep <- keep & passes_min_sample_threshold(dat, n_col)
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

seed_sem_col_for <- function(y_col, dat) {
  candidate <- paste0(y_col, "_seed_sem")
  legacy_candidate <- paste0(y_col, "_seed_sd")
  if (!is.null(dat) && candidate %in% names(dat)) {
    candidate
  } else if (!is.null(dat) && legacy_candidate %in% names(dat)) {
    legacy_candidate
  } else {
    NA_character_
  }
}

seed_log10_sem_col_for <- function(y_col, dat) {
  candidate <- paste0(y_col, "_seed_log10_sem")
  if (!is.null(dat) && candidate %in% names(dat)) {
    candidate
  } else {
    NA_character_
  }
}

log_display_values <- function(x) {
  x <- suppressWarnings(as.numeric(x))
  ifelse(is.finite(x) & x > 0, pmax(x, log_axis_floor), x)
}

y_values_with_seed_sem <- function(dat, y_col, log_y = FALSE) {
  y <- suppressWarnings(as.numeric(dat[[y_col]]))
  if (isTRUE(log_y)) {
    y <- log_display_values(y)
  }
  sem_col <- seed_sem_col_for(y_col, dat)
  if (!is.na(sem_col)) {
    if (isTRUE(log_y)) {
      log_sem_col <- seed_log10_sem_col_for(y_col, dat)
      if (!is.na(log_sem_col)) {
        log_sem_values <- suppressWarnings(as.numeric(dat[[log_sem_col]]))
        finite_sem <- is.finite(y) & y > 0 & is.finite(log_sem_values) & log_sem_values >= 0
        lows <- 10 ^ (log10(y[finite_sem]) - log_sem_values[finite_sem])
        highs <- 10 ^ (log10(y[finite_sem]) + log_sem_values[finite_sem])
        y <- c(y, lows, highs)
        return(y)
      }
    }
    sem_values <- suppressWarnings(as.numeric(dat[[sem_col]]))
    finite_sem <- is.finite(sem_values) & sem_values >= 0
    lows <- y[finite_sem] - sem_values[finite_sem]
    highs <- y[finite_sem] + sem_values[finite_sem]
    if (isTRUE(log_y)) {
      lows <- pmax(lows[lows > 0], log_axis_floor)
      highs <- pmax(highs[highs > 0], log_axis_floor)
    }
    y <- c(y, lows, highs)
  }
  y
}

draw_seed_error_bars <- function(x, y, sem_values, cols, log_y = FALSE, log_sem_values = NULL) {
  x <- suppressWarnings(as.numeric(x))
  y <- suppressWarnings(as.numeric(y))
  sem_values <- suppressWarnings(as.numeric(sem_values))
  keep <- is.finite(x) & is.finite(y) & is.finite(sem_values) & sem_values > 0
  if (isTRUE(log_y)) {
    y <- log_display_values(y)
    if (!is.null(log_sem_values)) {
      log_sem_values <- suppressWarnings(as.numeric(log_sem_values))
      keep <- is.finite(x) & is.finite(y) & y > 0 &
        is.finite(log_sem_values) & log_sem_values > 0
      if (!any(keep)) {
        return(invisible(NULL))
      }
      low <- 10 ^ (log10(y[keep]) - log_sem_values[keep])
      high <- 10 ^ (log10(y[keep]) + log_sem_values[keep])
      arrows(
        x0 = x[keep],
        y0 = low,
        x1 = x[keep],
        y1 = high,
        code = 3,
        angle = 90,
        length = 0.025,
        col = grDevices::adjustcolor(cols[keep], alpha.f = 0.65),
        lwd = 0.8
      )
      return(invisible(NULL))
    }
  }
  if (!any(keep)) {
    return(invisible(NULL))
  }
  low <- y[keep] - sem_values[keep]
  high <- y[keep] + sem_values[keep]
  if (isTRUE(log_y)) {
    low <- pmax(low, log_axis_floor)
    high <- pmax(high, log_axis_floor)
  }
  arrows(
    x0 = x[keep],
    y0 = low,
    x1 = x[keep],
    y1 = high,
    code = 3,
    angle = 90,
    length = 0.025,
    col = grDevices::adjustcolor(cols[keep], alpha.f = 0.65),
    lwd = 0.8
  )
  invisible(NULL)
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
  plot_variant <- if (identical(color_mode, "sigma")) {
    "sigma_color"
  } else if (isTRUE(facet_sigma)) {
    "sigma_panels"
  } else {
    "all_sigmas"
  }
  axis_prefix <- axis_plot_file_prefix(x_col, y_col, plot_variant)
  if (!should_write_revisit_axis_plot(axis_prefix)) {
    return(invisible(NULL))
  }
  path <- safe_png_path(axis_prefix)
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
  par(mar = panel_margins(top = if (isTRUE(facet_sigma) && n_panels > 1L) panel_title_margin_lines else panel_top_margin_lines))
  apply_panel_text_style()
  x_is_timestep <- is_timestep_axis_col(x_col) && !isTRUE(log_x)
  x_limits <- if (isTRUE(x_is_timestep)) {
    timestep_x_limits(plot_data[[x_col]])
  } else if (isTRUE(log_x)) {
    expand_log_range(log_display_values(plot_data[[x_col]]), pad = 0.05)
  } else {
    expand_range(plot_data[[x_col]], pad = 0.05)
  }
  y_limits <- if (isTRUE(log_y)) {
    expand_log_range(y_values_with_seed_sem(plot_data, y_col, log_y = TRUE), pad = 0.05)
  } else {
    expand_range(y_values_with_seed_sem(plot_data, y_col, log_y = FALSE), pad = 0.05)
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
      piece_x <- if (isTRUE(log_x)) log_display_values(piece[[x_col]]) else piece[[x_col]]
      piece_y <- if (isTRUE(log_y)) log_display_values(piece[[y_col]]) else piece[[y_col]]
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
        xaxt = if (isTRUE(x_is_timestep)) "n" else "s",
        log = log_arg_for_axes(log_x, log_y)
      )
      if (isTRUE(x_is_timestep)) {
        axis(1, at = timestep_x_ticks(plot_data[[x_col]]))
      }
      grid()
      sem_col <- seed_sem_col_for(y_col, piece)
      if (!is.na(sem_col)) {
        log_sem_col <- if (isTRUE(log_y)) seed_log10_sem_col_for(y_col, piece) else NA_character_
        draw_seed_error_bars(
          piece_x,
          piece_y,
          piece[[sem_col]],
          point_cols,
          log_y = log_y,
          log_sem_values = if (!is.na(log_sem_col)) piece[[log_sem_col]] else NULL
        )
      }
      draw_model_points(
        piece_x,
        piece_y,
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
  axis_prefix <- axis_plot_file_prefix(x_col, y_col, "sigma_panels")
  if (!should_write_revisit_axis_plot(axis_prefix)) {
    return(invisible(NULL))
  }
  path <- safe_png_path(axis_prefix)
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
  par(mar = panel_margins(top = if (n_sigma > 1L) panel_title_margin_lines else panel_top_margin_lines))
  apply_panel_text_style()
  x_limits <- if (!is.null(axis_spec)) {
    axis_spec$limits
  } else {
    expand_range(plot_data[[x_col]], pad = 0.06)
  }
  y_limits <- if (isTRUE(log_y)) {
    expand_log_range(y_values_with_seed_sem(plot_data, y_col, log_y = TRUE), pad = 0.06)
  } else {
    expand_range(y_values_with_seed_sem(plot_data, y_col, log_y = FALSE), pad = 0.06)
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
          piece_y <- if (isTRUE(log_y)) log_display_values(piece[[y_col]]) else piece[[y_col]]
          lines(
            piece[[x_col]],
            piece_y,
            type = "l",
            lwd = 1.4,
            col = piece_col
          )
          sem_col <- seed_sem_col_for(y_col, piece)
          if (!is.na(sem_col)) {
            log_sem_col <- if (isTRUE(log_y)) seed_log10_sem_col_for(y_col, piece) else NA_character_
            draw_seed_error_bars(
              piece[[x_col]],
              piece_y,
              piece[[sem_col]],
              rep(piece_col, nrow(piece)),
              log_y = log_y,
              log_sem_values = if (!is.na(log_sem_col)) piece[[log_sem_col]] else NULL
            )
          }
          draw_model_points(
            piece[[x_col]],
            piece_y,
            rep(model_value, nrow(piece)),
            rep(piece_col, nrow(piece)),
            cex = 1
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
  log_y = FALSE,
  x_file_col = "timestep"
) {
  axis_prefix <- axis_plot_file_prefix(x_file_col, y_col, "sigma_panels")
  if (!should_write_revisit_axis_plot(axis_prefix)) {
    return(invisible(NULL))
  }
  path <- safe_png_path(axis_prefix)
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
  par(mar = panel_margins(top = if (n_sigma > 1L) panel_title_margin_lines else panel_top_margin_lines))
  apply_panel_text_style()
  x_values <- suppressWarnings(as.numeric(plot_data$timestep))
  x_limits <- timestep_x_limits(x_values)
  x_ticks <- timestep_x_ticks(x_values)
  y_limits <- if (isTRUE(log_y)) {
    expand_log_range(y_values_with_seed_sem(plot_data, y_col, log_y = TRUE), pad = 0.06)
  } else {
    expand_range(y_values_with_seed_sem(plot_data, y_col, log_y = FALSE), pad = 0.06)
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
          piece_y <- if (isTRUE(log_y)) log_display_values(piece[[y_col]]) else piece[[y_col]]
          lines(
            piece$timestep,
            piece_y,
            type = "l",
            lwd = 1.4,
            col = piece_col
          )
          sem_col <- seed_sem_col_for(y_col, piece)
          if (!is.na(sem_col)) {
            log_sem_col <- if (isTRUE(log_y)) seed_log10_sem_col_for(y_col, piece) else NA_character_
            draw_seed_error_bars(
              piece$timestep,
              piece_y,
              piece[[sem_col]],
              rep(piece_col, nrow(piece)),
              log_y = log_y,
              log_sem_values = if (!is.na(log_sem_col)) piece[[log_sem_col]] else NULL
            )
          }
          draw_model_points(
            piece$timestep,
            piece_y,
            rep(model_value, nrow(piece)),
            rep(piece_col, nrow(piece)),
            cex = 1
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

plot_metric_by_timestep_and_total_timestep <- function(
  summary_data,
  y_col,
  ylab,
  file_prefix,
  log_y = FALSE
) {
  axis_prefix <- axis_plot_file_prefix("timestep", y_col, "by_total_timestep_sigma_panels")
  if (!should_write_revisit_axis_plot(axis_prefix)) {
    return(invisible(NULL))
  }
  path <- safe_png_path(axis_prefix)
  required_cols <- c("timestep", "timestep_before_stop")
  if (!all(required_cols %in% names(summary_data))) {
    return(invisible(NULL))
  }
  plot_data <- filter_plot_data(summary_data, "timestep", y_col, log_x = FALSE, log_y = log_y, file_prefix = file_prefix)
  plot_data <- plot_data[
    is.finite(suppressWarnings(as.numeric(plot_data$timestep_before_stop))),
    ,
    drop = FALSE
  ]
  if (nrow(plot_data) == 0) {
    return(invisible(NULL))
  }
  total_levels <- sort(unique(suppressWarnings(as.numeric(plot_data$timestep_before_stop))))
  total_levels <- total_levels[is.finite(total_levels)]
  if (length(total_levels) == 0L) {
    return(invisible(NULL))
  }
  n_sigma <- max(1L, length(sigma_levels))
  n_total <- length(total_levels)
  model_levels <- if ("model" %in% names(plot_data)) ordered_model_levels(plot_data$model) else "VAE"
  open_panel_png(path, n_cols = n_sigma, n_rows = n_total, legend_fraction = legend_panel_fraction)
  old_par <- par(no.readonly = TRUE)
  panel_matrix <- matrix(seq_len(n_sigma * n_total), nrow = n_total, ncol = n_sigma, byrow = TRUE)
  layout(cbind(panel_matrix, rep(n_sigma * n_total + 1L, n_total)),
    widths = c(rep(1, n_sigma), legend_panel_fraction)
  )
  par(mar = panel_margins(top = if (n_sigma > 1L) panel_title_margin_lines else panel_top_margin_lines))
  apply_panel_text_style()
  x_values <- suppressWarnings(as.numeric(plot_data$timestep))
  x_limits <- timestep_x_limits(x_values)
  x_ticks <- timestep_x_ticks(x_values)
  y_limits <- if (isTRUE(log_y)) {
    expand_log_range(y_values_with_seed_sem(plot_data, y_col, log_y = TRUE), pad = 0.06)
  } else {
    expand_range(y_values_with_seed_sem(plot_data, y_col, log_y = FALSE), pad = 0.06)
  }
  for (total_value in total_levels) {
    total_data <- plot_data[
      abs(suppressWarnings(as.numeric(plot_data$timestep_before_stop)) - total_value) < 1e-8,
      ,
      drop = FALSE
    ]
    total_label <- format_plot_values(total_value)
    for (sigma_value in sigma_levels) {
      panel_data <- filter_sigma_rows(total_data, sigma_value)
      plot(
        NA,
        xlim = x_limits,
        ylim = y_limits,
        xlab = "Pre-stop observation timestep",
        ylab = ylab,
        main = if (n_sigma > 1L) {
          sprintf("total t=%s | %s", total_label, sigma_panel_title(sigma_value))
        } else {
          sprintf("total t=%s", total_label)
        },
        xaxt = "n",
        log = log_arg_for_axes(FALSE, log_y)
      )
      axis(1, at = x_ticks)
      grid()
      for (model_value in model_levels) {
        model_data <- if ("model" %in% names(panel_data)) {
          filter_model_rows(panel_data, model_value)
        } else {
          panel_data
        }
        for (opportunity_value in opportunity_levels) {
          for (beta_value in beta_levels) {
            piece <- model_data[
              parameter_value_matches(model_data$beta, beta_value) &
                parameter_value_matches(model_data$opportunity, opportunity_value),
              ,
              drop = FALSE
            ]
            if (nrow(piece) == 0) {
              next
            }
            piece <- piece[order(piece$timestep), , drop = FALSE]
            piece_col <- line_color_for(beta_value, opportunity_value, model_value)
            piece_y <- if (isTRUE(log_y)) log_display_values(piece[[y_col]]) else piece[[y_col]]
            lines(
              piece$timestep,
              piece_y,
              type = "l",
              lwd = 1.25,
              lty = if (identical(as.character(model_value), "Optimal")) 2 else 1,
              col = piece_col
            )
            sem_col <- seed_sem_col_for(y_col, piece)
            if (!is.na(sem_col)) {
              log_sem_col <- if (isTRUE(log_y)) seed_log10_sem_col_for(y_col, piece) else NA_character_
              draw_seed_error_bars(
                piece$timestep,
                piece_y,
                piece[[sem_col]],
                rep(piece_col, nrow(piece)),
                log_y = log_y,
                log_sem_values = if (!is.na(log_sem_col)) piece[[log_sem_col]] else NULL
              )
            }
            draw_model_points(
              piece$timestep,
              piece_y,
              rep(model_value, nrow(piece)),
              rep(piece_col, nrow(piece)),
              cex = 0.95
            )
          }
        }
      }
    }
  }
  par(mar = c(0, 0, 0, 0))
  plot_parameter_legend(model_levels)
  par(old_par)
  dev.off()
  message(sprintf("Saved %s", path))
}

heatmap_half_width <- function(values, default = 0.5) {
  values <- sort(unique(suppressWarnings(as.numeric(values))))
  values <- values[is.finite(values)]
  if (length(values) < 2L) {
    return(default)
  }
  diffs <- diff(values)
  diffs <- diffs[is.finite(diffs) & diffs > 0]
  if (length(diffs) == 0) {
    return(default)
  }
  min(diffs) / 2
}

heatmap_color_for <- function(z, zlim, palette_cols) {
  z <- suppressWarnings(as.numeric(z))
  out <- rep(NA_character_, length(z))
  keep <- is.finite(z)
  if (!any(keep)) {
    return(out)
  }
  if (!all(is.finite(zlim)) || abs(diff(zlim)) < 1e-12) {
    idx <- rep(ceiling(length(palette_cols) / 2), sum(keep))
  } else {
    scaled <- (z[keep] - zlim[[1]]) / diff(zlim)
    idx <- pmax(1L, pmin(length(palette_cols), floor(scaled * (length(palette_cols) - 1L)) + 1L))
  }
  out[keep] <- palette_cols[idx]
  out
}

draw_heatmap_legend <- function(zlim, palette_cols, label) {
  plot.new()
  plot.window(xlim = c(0, 1), ylim = zlim)
  n <- length(palette_cols)
  y_edges <- seq(zlim[[1]], zlim[[2]], length.out = n + 1L)
  for (i in seq_len(n)) {
    rect(0.2, y_edges[[i]], 0.6, y_edges[[i + 1L]], col = palette_cols[[i]], border = NA)
  }
  axis(4, las = 1, cex.axis = 1)
  mtext(label, side = 4, line = 2.4, cex = 1)
}

draw_reward_context_heatmap_panel <- function(
  panel_data,
  value_col,
  zlim,
  palette_cols,
  main = "",
  x_values = NULL,
  y_values = NULL
) {
  x <- suppressWarnings(as.numeric(panel_data$mean_other_path_actual_reward))
  y <- suppressWarnings(as.numeric(panel_data$first_observed_path_actual_reward))
  z <- suppressWarnings(as.numeric(panel_data[[value_col]]))
  if (is.null(x_values)) {
    x_values <- sort(unique(x[is.finite(x)]))
  } else {
    x_values <- sort(unique(suppressWarnings(as.numeric(x_values))))
    x_values <- x_values[is.finite(x_values)]
  }
  if (is.null(y_values)) {
    y_values <- sort(unique(y[is.finite(y)]))
  } else {
    y_values <- sort(unique(suppressWarnings(as.numeric(y_values))))
    y_values <- y_values[is.finite(y_values)]
  }
  xlim <- if (length(x_values) > 0L) {
    range(x_values) + c(-1, 1) * heatmap_half_width(x_values)
  } else {
    c(-1, 1)
  }
  ylim <- if (length(y_values) > 0L) {
    range(y_values) + c(-1, 1) * heatmap_half_width(y_values)
  } else {
    c(-1, 1)
  }
  plot(
    NA,
    xlim = xlim,
    ylim = ylim,
    xlab = "Mean actual reward of other paths",
    ylab = "Actual reward of first observed path",
    xaxt = "n",
    yaxt = "n",
    main = main
  )
  if (length(x_values) > 0L) {
    axis(1, at = x_values, labels = format_plot_values(x_values), las = 2, cex.axis = 1)
  }
  if (length(y_values) > 0L) {
    axis(2, at = y_values, labels = format_plot_values(y_values), las = 1, cex.axis = 1)
  }
  grid()
  if (nrow(panel_data) == 0) {
    return(invisible(NULL))
  }
  x_half <- heatmap_half_width(x_values)
  y_half <- heatmap_half_width(y_values)
  cols <- heatmap_color_for(z, zlim, palette_cols)
  keep <- is.finite(x) & is.finite(y) & is.finite(z) & !is.na(cols)
  if (any(keep)) {
    rect(
      xleft = x[keep] - x_half,
      ybottom = y[keep] - y_half,
      xright = x[keep] + x_half,
      ytop = y[keep] + y_half,
      col = cols[keep],
      border = "white",
      lwd = 0.35
    )
  }
  invisible(NULL)
}

heatmap_combo_title <- function(combo) {
  sprintf(
    "%s | sigma %s\nbeta %s | opp %s",
    as.character(combo$model[[1]]),
    format_plot_values(combo$sigma[[1]]),
    format_plot_values(combo$beta[[1]]),
    format_plot_values(combo$opportunity[[1]])
  )
}

plot_reward_context_heatmaps <- function(
  summary_data,
  value_col,
  value_label,
  file_prefix,
  include_timestep = TRUE,
  palette = "Viridis"
) {
  if (isTRUE(selected_revisit_plots_only)) {
    return(invisible(NULL))
  }
  if (is.null(summary_data) || nrow(summary_data) == 0 || !value_col %in% names(summary_data)) {
    warning(sprintf("Missing heatmap data for %s; skipping.", file_prefix))
    return(invisible(NULL))
  }
  count_col <- paste0(value_col, "_n")
  if (!count_col %in% names(summary_data)) {
    count_col <- if ("n" %in% names(summary_data)) "n" else NA_character_
  }
  plot_data <- summary_data
  plot_data[[value_col]] <- suppressWarnings(as.numeric(plot_data[[value_col]]))
  keep <- is.finite(plot_data[[value_col]])
  if (!is.na(count_col)) {
    keep <- keep & passes_min_sample_threshold(plot_data, count_col)
  }
  plot_data <- plot_data[keep, , drop = FALSE]
  if (nrow(plot_data) == 0) {
    warning(sprintf(
      "No heatmap cells remained after finite-value and VAE minimum-sample filtering for %s; skipping.",
      file_prefix
    ))
    return(invisible(NULL))
  }
  zlim <- range(plot_data[[value_col]], na.rm = TRUE)
  if (!all(is.finite(zlim)) || abs(diff(zlim)) < 1e-12) {
    zlim <- zlim + c(-0.5, 0.5)
  }
  palette_cols <- grDevices::hcl.colors(96, palette = palette)
  combo_cols <- c("model", "sigma", "beta", "opportunity")
  combos <- unique(plot_data[, combo_cols, drop = FALSE])
  combos <- combos[order(combos$model, suppressWarnings(as.numeric(combos$sigma)), suppressWarnings(as.numeric(combos$beta)), suppressWarnings(as.numeric(combos$opportunity))), , drop = FALSE]
  x_values_all <- sort(unique(suppressWarnings(as.numeric(plot_data$mean_other_path_actual_reward))))
  x_values_all <- x_values_all[is.finite(x_values_all)]
  y_values_all <- sort(unique(suppressWarnings(as.numeric(plot_data$first_observed_path_actual_reward))))
  y_values_all <- y_values_all[is.finite(y_values_all)]
  timestep_levels <- if (isTRUE(include_timestep) && "timestep" %in% names(plot_data)) {
    sort(unique(suppressWarnings(as.numeric(plot_data$timestep))))
  } else {
    NA_real_
  }
  timestep_levels <- timestep_levels[is.finite(timestep_levels) | is.na(timestep_levels)]
  if (length(timestep_levels) == 0L) {
    warning(sprintf("No timestep levels for %s; skipping.", file_prefix))
    return(invisible(NULL))
  }
  n_combos <- nrow(combos)
  n_cols <- max(1L, min(5L, ceiling(sqrt(n_combos))))
  n_rows <- ceiling(n_combos / n_cols)
  for (timestep_value in timestep_levels) {
    timestep_suffix <- if (isTRUE(include_timestep) && is.finite(timestep_value)) {
      paste0("timestep_", short_num_label(timestep_value))
    } else {
      "all_param_combos"
    }
    path <- safe_png_path(axis_plot_file_prefix(
      "mean_other_path_actual_reward",
      value_col,
      paste(file_prefix, timestep_suffix, sep = "_")
    ))
    open_panel_png(path, n_cols = n_cols, n_rows = n_rows, legend_fraction = 0.45)
    old_par <- par(no.readonly = TRUE)
    panel_ids <- matrix(seq_len(n_rows * n_cols), nrow = n_rows, ncol = n_cols, byrow = TRUE)
    layout(cbind(panel_ids, rep(n_rows * n_cols + 1L, n_rows)), widths = c(rep(1, n_cols), 0.45))
    par(mar = panel_margins(top = 3.2, bottom = 4.8))
    apply_panel_text_style()
    for (combo_i in seq_len(n_rows * n_cols)) {
      if (combo_i > nrow(combos)) {
        plot.new()
        next
      }
      combo <- combos[combo_i, , drop = FALSE]
      combo_data <- plot_data[
        as.character(plot_data$model) == as.character(combo$model[[1]]) &
          sigma_value_matches(plot_data$sigma, combo$sigma[[1]]) &
          parameter_value_matches(plot_data$beta, combo$beta[[1]]) &
          parameter_value_matches(plot_data$opportunity, combo$opportunity[[1]]),
        ,
        drop = FALSE
      ]
      panel_data <- if (isTRUE(include_timestep) && is.finite(timestep_value)) {
        combo_data[abs(suppressWarnings(as.numeric(combo_data$timestep)) - timestep_value) < 1e-8, , drop = FALSE]
      } else {
        combo_data
      }
      draw_reward_context_heatmap_panel(
        panel_data,
        value_col,
        zlim,
        palette_cols,
        main = heatmap_combo_title(combo),
        x_values = x_values_all,
        y_values = y_values_all
      )
    }
    par(mar = c(4.8, 0.2, 3.2, 3.8))
    draw_heatmap_legend(zlim, palette_cols, value_label)
    par(old_par)
    dev.off()
    message(sprintf("Saved %s", path))
  }
  invisible(NULL)
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
  actual_reward_mat <- actual_reward_matrix_for_trials(trial_data, actual_lookup)
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
  path_reward_mat <- path_reward_matrix_from_node_rewards(actual_reward_mat)
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
  kl_paid_at_stop_timestep <- rep(NA_real_, nrow(trial_data))
  stop_observation_timestep <- suppressWarnings(as.integer(round(timestep_before_stop)))
  if (
    exists("kl_mat", inherits = TRUE) &&
      exists("kl_timesteps", inherits = TRUE) &&
      nrow(kl_mat) == nrow(trial_data) &&
      length(kl_timesteps) > 0L
  ) {
    stop_kl_col <- match(stop_observation_timestep, kl_timesteps)
    valid_stop_kl <- !is.na(stop_kl_col) &
      is.finite(timestep_before_stop) &
      timestep_before_stop > 0
    if (any(valid_stop_kl)) {
      stop_kl_rows <- which(valid_stop_kl)
      kl_paid_at_stop_timestep[stop_kl_rows] <- kl_mat[cbind(stop_kl_rows, stop_kl_col[stop_kl_rows])]
    }
  }

  row_mean_excluding_path <- function(reward_matrix, exclude_path) {
    out <- rep(NA_real_, nrow(reward_matrix))
    exclude_path <- suppressWarnings(as.integer(exclude_path))
    valid <- !is.na(exclude_path) & exclude_path >= 1L & exclude_path <= ncol(reward_matrix)
    if (any(valid)) {
      valid_rows <- which(valid)
      excluded_rewards <- reward_matrix[cbind(valid_rows, exclude_path[valid_rows])]
      out[valid_rows] <- (rowSums(reward_matrix[valid_rows, , drop = FALSE]) - excluded_rewards) /
        (ncol(reward_matrix) - 1L)
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
    first_path_col <- max.col(ifelse(finite_path_mat, ncol(path_mat) - path_col_indices + 1L, 0),
      ties.method = "first"
    )
    last_path_col <- max.col(ifelse(finite_path_mat, path_col_indices, 0), ties.method = "first")
    has_path <- rowSums(finite_path_mat) > 0
    first_path <- rep(NA_real_, nrow(path_mat))
    last_path <- rep(NA_real_, nrow(path_mat))
    first_path[has_path] <- path_mat[cbind(which(has_path), first_path_col[has_path])]
    last_path[has_path] <- path_mat[cbind(which(has_path), last_path_col[has_path])]
  } else {
    first_path <- rep(NA_real_, nrow(trial_data))
    last_path <- rep(NA_real_, nrow(trial_data))
  }
  first_path_int <- suppressWarnings(as.integer(first_path))
  last_path_int <- suppressWarnings(as.integer(last_path))
  first_observed_path_reward <- rep(NA_real_, nrow(trial_data))
  valid_first_path <- !is.na(first_path_int) & first_path_int >= 1L & first_path_int <= ncol(path_reward_mat)
  if (any(valid_first_path)) {
    first_rows <- which(valid_first_path)
    first_observed_path_reward[first_rows] <- path_reward_mat[cbind(first_rows, first_path_int[first_rows])]
  }
  last_visited_path_reward <- rep(NA_real_, nrow(trial_data))
  valid_last_path <- !is.na(last_path_int) & last_path_int >= 1L & last_path_int <= ncol(path_reward_mat)
  if (any(valid_last_path)) {
    last_rows <- which(valid_last_path)
    last_visited_path_reward[last_rows] <- path_reward_mat[cbind(last_rows, last_path_int[last_rows])]
  }
  other_mean_after_first_observation <- row_mean_excluding_path(path_reward_mat, first_path_int)
  other_mean_after_last_visit <- row_mean_excluding_path(path_reward_mat, last_path_int)
  path_1_minus_mean_other_path_raw <- path_1_reward - path_1_mean_other_reward
  first_observed_minus_mean_other_path_raw <- first_observed_path_reward - other_mean_after_first_observation
  last_visited_minus_mean_other_path_raw <- last_visited_path_reward - other_mean_after_last_visit
  first_path_visit_mat <- if (ncol(path_mat) > 0L) {
    first_path_matrix <- matrix(first_path, nrow = nrow(path_mat), ncol = ncol(path_mat))
    (path_mat == first_path_matrix) & pre_stop_observation_mat
  } else {
    matrix(FALSE, nrow = nrow(trial_data), ncol = 0L)
  }
  first_observed_path_visits <- rowSums(first_path_visit_mat, na.rm = TRUE)
  chosen_path <- coerce_chosen_path_to_one_based(trial_data$chosen_path, trial_data$model)
  trial_metrics <- data.frame(
    model = trial_data$model,
    sigma = trial_data$sigma,
    beta = trial_data$beta,
    opportunity = trial_data$opportunity,
    seed = trial_data$seed,
    path_reward_difference = path_1_minus_mean_other_path_raw,
    absolute_path_reward_difference = abs(path_1_minus_mean_other_path_raw),
    first_observed_path_reward_difference = first_observed_minus_mean_other_path_raw,
    absolute_first_observed_path_reward_difference = abs(first_observed_minus_mean_other_path_raw),
    last_visited_path_reward_difference = last_visited_minus_mean_other_path_raw,
    path_1_minus_mean_other_path = bin_mean_difference(path_1_minus_mean_other_path_raw),
    absolute_path_1_minus_mean_other_path = bin_mean_difference(abs(path_1_minus_mean_other_path_raw)),
    first_observed_minus_mean_other_path = bin_mean_difference(first_observed_minus_mean_other_path_raw),
    absolute_first_observed_minus_mean_other_path = bin_mean_difference(abs(first_observed_minus_mean_other_path_raw)),
    last_visited_minus_mean_other_path = bin_mean_difference(last_visited_minus_mean_other_path_raw),
    proportion_timesteps_path1 = ifelse(observations > 0, path_1_visits / observations, NA_real_),
    proportion_timesteps_first_observed_path = ifelse(observations > 0, first_observed_path_visits / observations, NA_real_),
    last_visited_path_chosen = ifelse(is.finite(last_path) & !is.na(chosen_path), as.numeric(last_path == chosen_path), NA_real_),
    choose_path1 = ifelse(!is.na(chosen_path), as.numeric(chosen_path == 1L), NA_real_),
    choose_first_observed_path = ifelse(is.finite(first_path) & !is.na(chosen_path), as.numeric(first_path == chosen_path), NA_real_),
    node_reward_difference = path_1_minus_mean_other_path_raw,
    absolute_node_reward_difference = abs(path_1_minus_mean_other_path_raw),
    first_observed_node_reward_difference = first_observed_minus_mean_other_path_raw,
    absolute_first_observed_node_reward_difference = abs(first_observed_minus_mean_other_path_raw),
    last_visited_reward_difference = last_visited_minus_mean_other_path_raw,
    node_1_minus_mean_other_node = bin_mean_difference(path_1_minus_mean_other_path_raw),
    absolute_node_1_minus_mean_other_node = bin_mean_difference(abs(path_1_minus_mean_other_path_raw)),
    first_observed_minus_mean_other_node = bin_mean_difference(first_observed_minus_mean_other_path_raw),
    absolute_first_observed_minus_mean_other_node = bin_mean_difference(abs(first_observed_minus_mean_other_path_raw)),
    last_visited_minus_mean_other_node = bin_mean_difference(last_visited_minus_mean_other_path_raw),
    proportion_timesteps_node1 = ifelse(observations > 0, path_1_visits / observations, NA_real_),
    proportion_timesteps_first_observed_node = ifelse(observations > 0, first_observed_path_visits / observations, NA_real_),
    last_visited_node_chosen = ifelse(is.finite(last_path) & !is.na(chosen_path), as.numeric(last_path == chosen_path), NA_real_),
    choose_node1 = ifelse(!is.na(chosen_path), as.numeric(chosen_path == 1L), NA_real_),
    choose_first_observed_node = ifelse(is.finite(first_path) & !is.na(chosen_path), as.numeric(first_path == chosen_path), NA_real_),
    observations_before_stop = observations,
    stop_decision_timestep = stop_timestep,
    timestep_before_stop = timestep_before_stop,
    kl_paid_total = suppressWarnings(as.numeric(trial_data$kl_paid_total)),
    kl_paid_per_stop_timestep = ifelse(
      is.finite(timestep_before_stop) & timestep_before_stop > 0,
      suppressWarnings(as.numeric(trial_data$kl_paid_total)) / timestep_before_stop,
      NA_real_
    ),
    kl_paid_per_pre_stop_timestep = ifelse(
      is.finite(observations) & observations > 1,
      suppressWarnings(as.numeric(trial_data$kl_paid_total)) / (observations - 1),
      NA_real_
    ),
    kl_paid_at_stop_timestep = kl_paid_at_stop_timestep,
    kl_paid_at_stop_timestep_after_first_continue = ifelse(
      is.finite(observations) & observations > 1,
      kl_paid_at_stop_timestep,
      NA_real_
    ),
    stringsAsFactors = FALSE
  )
  trial_metrics[
    is.finite(trial_metrics$path_1_minus_mean_other_path) |
      is.finite(trial_metrics$absolute_path_1_minus_mean_other_path) |
      is.finite(trial_metrics$first_observed_minus_mean_other_path) |
      is.finite(trial_metrics$absolute_first_observed_minus_mean_other_path) |
      is.finite(trial_metrics$last_visited_minus_mean_other_path) |
      is.finite(trial_metrics$node_reward_difference) |
      is.finite(trial_metrics$absolute_node_reward_difference) |
      is.finite(trial_metrics$first_observed_node_reward_difference) |
      is.finite(trial_metrics$absolute_first_observed_node_reward_difference) |
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
  difference_value_cols <- c(
    "proportion_timesteps_path1",
    "proportion_timesteps_first_observed_path",
    "last_visited_path_chosen",
    "choose_path1",
    "choose_first_observed_path",
    "observations_before_stop",
    "stop_decision_timestep",
    "timestep_before_stop",
    "kl_paid_total",
    "kl_paid_per_stop_timestep",
    "kl_paid_per_pre_stop_timestep",
    "kl_paid_at_stop_timestep",
    "kl_paid_at_stop_timestep_after_first_continue"
  )
  out <- aggregate_means_by(
    trial_metrics,
    group_cols = group_cols,
    value_cols = difference_value_cols
  )
  count_data <- count_rows_by(trial_metrics, group_cols)
  out <- merge(out, count_data, by = group_cols, all.x = TRUE)
  for (value_col in difference_value_cols) {
    finite_rows <- is.finite(suppressWarnings(as.numeric(trial_metrics[[value_col]])))
    if (!any(finite_rows)) {
      next
    }
    value_count_data <- count_rows_by(
      trial_metrics[finite_rows, , drop = FALSE],
      group_cols,
      count_name = paste0(value_col, "_n")
    )
    out <- merge(out, value_count_data, by = group_cols, all.x = TRUE)
  }
  add_seed_sem_by(
    out,
    trial_metrics,
    group_cols = group_cols,
    value_cols = difference_value_cols
  )
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
  ylab = "Average KL paid across timesteps",
  file_prefix = "revisit_kl_paid_vs_average_normalized_chosen_path_reward_by_sigma",
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
  "terminal_choice_entropy_combined_reached" %in% names(entropy_average_summary) &&
    any(is.finite(suppressWarnings(as.numeric(entropy_average_summary$terminal_choice_entropy_combined_reached))))
) {
  "terminal_choice_entropy_combined_reached"
} else {
  "terminal_choice_entropy"
}
entropy_summary_label <- if (identical(entropy_summary_col, "terminal_choice_entropy_combined_reached")) {
  "Average stop-choice entropy\nacross reached timesteps"
} else {
  "Average terminal choice\nentropy at stop"
}
has_entropy_data <- entropy_summary_col %in% names(entropy_average_summary) &&
  any(is.finite(suppressWarnings(as.numeric(entropy_average_summary[[entropy_summary_col]]))))
if (has_entropy_data) {
  plot_summary_scatter(
    entropy_average_summary,
    x_col = "kl_paid_total",
    y_col = entropy_summary_col,
    xlab = "Average KL paid\nacross timesteps (log)",
    ylab = entropy_summary_label,
    file_prefix = "revisit_average_terminal_choice_entropy_vs_log_kl_paid_by_sigma",
    log_x = TRUE,
    facet_sigma = TRUE,
    color_mode = "parameter",
    model_layout = "overlay"
  )
  plot_summary_scatter(
    entropy_average_summary,
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
    entropy_average_summary,
    x_col = "kl_paid_total",
    y_col = entropy_summary_col,
    xlab = "Average KL paid\nacross timesteps (log)",
    ylab = entropy_summary_label,
    file_prefix = "revisit_average_terminal_choice_entropy_vs_log_kl_paid_sigma_color",
    log_x = TRUE,
    facet_sigma = FALSE,
    color_mode = "sigma",
    model_layout = "overlay"
  )
  plot_summary_scatter(
    entropy_average_summary,
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

is_default_two_node_tree <- tree_size == 2L && !nzchar(tree_config)
if (is_default_two_node_tree) {
  has_max_terminal_prob_data <- "terminal_choice_prob_max_combined_reached" %in% names(entropy_average_summary) &&
    any(is.finite(suppressWarnings(as.numeric(entropy_average_summary$terminal_choice_prob_max_combined_reached))))
  if (has_max_terminal_prob_data) {
    plot_summary_scatter(
      entropy_average_summary,
      x_col = "timestep_before_stop",
      y_col = "terminal_choice_prob_max_combined_reached",
      xlab = "Average timestep before stopping",
      ylab = "Average max P(choose path 1, path 2)\nacross reached timesteps",
      file_prefix = "revisit_average_terminal_choice_max_probability_vs_average_timestep_before_stop_by_sigma",
      facet_sigma = TRUE,
      color_mode = "parameter",
      model_layout = "overlay"
    )
  }
  has_min_terminal_prob_data <- "terminal_choice_prob_min_combined_reached" %in% names(entropy_average_summary) &&
    any(is.finite(suppressWarnings(as.numeric(entropy_average_summary$terminal_choice_prob_min_combined_reached))))
  if (has_min_terminal_prob_data) {
    plot_summary_scatter(
      entropy_average_summary,
      x_col = "timestep_before_stop",
      y_col = "terminal_choice_prob_min_combined_reached",
      xlab = "Average timestep before stopping",
      ylab = "Average min P(choose path 1, path 2)\nacross reached timesteps",
      file_prefix = "revisit_average_terminal_choice_min_probability_vs_average_timestep_before_stop_by_sigma",
      facet_sigma = TRUE,
      color_mode = "parameter",
      model_layout = "overlay"
    )
  }
}

plot_metric_by_timestep(
  pre_stop_timestep_summary,
  y_col = "kl_paid_at_timestep",
  ylab = "Average carried-forward KL\nafter timestep",
  file_prefix = "revisit_pre_stop_timestep_vs_kl_paid_at_timestep_by_sigma"
)
plot_metric_by_timestep_and_total_timestep(
  pre_stop_timestep_by_total_summary,
  y_col = "kl_paid_at_timestep",
  ylab = "Average carried-forward KL\nafter timestep",
  file_prefix = "revisit_pre_stop_timestep_vs_kl_paid_at_timestep_by_total_timestep_by_sigma"
)
if (nrow(first_timestep_kl_after_continue_summary) > 0) {
  plot_metric_by_difference(
    first_timestep_kl_after_continue_summary,
    y_col = "kl_paid_at_first_timestep_after_continue",
    ylab = "Average KL after first reward\ncontinued to another observation",
    file_prefix = "revisit_first_observed_reward_vs_kl_paid_at_first_timestep_after_continue_by_sigma",
    x_col = "first_observed_reward_t1",
    xlab = "Actual reward of first observed node"
  )
} else {
  warning(
    "No first-timestep KL rows where the model continued after the first reward; skipping first-reward KL plot."
  )
}
has_pre_stop_entropy_data <- nrow(pre_stop_timestep_entropy_summary) > 0 &&
  "terminal_binary_choice_entropy_at_timestep" %in% names(pre_stop_timestep_entropy_summary) &&
  any(is.finite(suppressWarnings(as.numeric(pre_stop_timestep_entropy_summary$terminal_binary_choice_entropy_at_timestep))))
if (has_pre_stop_entropy_data) {
  plot_metric_by_timestep(
    pre_stop_timestep_entropy_summary,
    y_col = "terminal_binary_choice_entropy_at_timestep",
    ylab = "Average stop-choice entropy\npath 1 vs path 2",
    file_prefix = "revisit_pre_stop_timestep_vs_terminal_choice_entropy_at_timestep_by_sigma"
  )
} else {
  warning(
    "No terminal choice probability columns found at pre-stop timesteps; skipping pre-stop terminal-choice entropy plot."
  )
}
has_strict_pre_stop_entropy_data <- nrow(pre_stop_timestep_entropy_strict_summary) > 0 &&
  "terminal_binary_choice_entropy_at_timestep" %in% names(pre_stop_timestep_entropy_strict_summary) &&
  any(is.finite(suppressWarnings(as.numeric(pre_stop_timestep_entropy_strict_summary$terminal_binary_choice_entropy_at_timestep))))
if (has_strict_pre_stop_entropy_data) {
  plot_metric_by_timestep(
    pre_stop_timestep_entropy_strict_summary,
    y_col = "terminal_binary_choice_entropy_at_timestep",
    ylab = "Average stop-choice entropy\nbefore stopping\npath 1 vs path 2",
    file_prefix = "revisit_strict_pre_stop_timestep_vs_terminal_choice_entropy_at_timestep_by_sigma",
    x_file_col = "strict_pre_stop_timestep"
  )
} else {
  warning(
    "No terminal choice probability columns found strictly before stopping; skipping strict pre-stop terminal-choice entropy plot."
  )
}
plot_heatmap_summary_set <- function(summary_set, prefix_suffix = "") {
  if (is.null(summary_set)) {
    return(invisible(NULL))
  }
  plot_reward_context_heatmaps(
    summary_set$terminal_entropy,
    value_col = "terminal_choice_entropy_at_timestep",
    value_label = "Terminal action-policy entropy",
    file_prefix = paste0(
      "terminal_action_policy_entropy_by_first_observed_and_mean_other_reward_heatmap",
      prefix_suffix
    ),
    include_timestep = TRUE,
    palette = "Viridis"
  )
  plot_reward_context_heatmaps(
    summary_set$kl_timestep,
    value_col = "kl_paid_at_timestep",
    value_label = "KL paid after continuing",
    file_prefix = paste0(
      "kl_paid_by_first_observed_and_mean_other_reward_heatmap",
      prefix_suffix
    ),
    include_timestep = TRUE,
    palette = "Plasma"
  )
  plot_reward_context_heatmaps(
    summary_set$timestep_before_stop,
    value_col = "timestep_before_stop",
    value_label = "Timestep before stopping",
    file_prefix = paste0(
      "timestep_before_stop_by_first_observed_and_mean_other_reward_heatmap",
      prefix_suffix
    ),
    include_timestep = FALSE,
    palette = "YlGnBu"
  )
  invisible(NULL)
}

plot_heatmap_summary_set(default_heatmap_summaries)
plot_heatmap_summary_set(disjoint_integer_heatmap_summaries, "_path_value_mean_other_integer")
plot_heatmap_summary_set(disjoint_bin2_heatmap_summaries, "_path_value_bin2")

node_difference_trial_metrics <- build_node_difference_trial_metrics(trial_data, node_mat)
if (nrow(node_difference_trial_metrics) > 0) {
  first_minus_other_path_label <- "First observed path reward\n- mean other path reward"
  abs_first_minus_other_path_label <- "|First observed path reward\n- mean other path reward|"
  last_minus_other_path_label <- "Last visited path reward\n- mean other path reward"
  node_difference_summary <- summarize_difference_metrics(
    node_difference_trial_metrics,
    "first_observed_minus_mean_other_path"
  )
  last_visited_difference_summary <- summarize_difference_metrics(
    node_difference_trial_metrics,
    "last_visited_minus_mean_other_path"
  )
  absolute_node_difference_summary <- summarize_difference_metrics(
    node_difference_trial_metrics,
    "absolute_first_observed_minus_mean_other_path"
  )
  plot_metric_by_difference(
    node_difference_summary,
    y_col = "proportion_timesteps_first_observed_path",
    ylab = "Proportion pre-stop timesteps\nvisiting first observed path",
    file_prefix = "revisit_first_observed_minus_mean_other_path_reward_vs_proportion_first_observed_path_visits_by_sigma",
    x_col = "first_observed_minus_mean_other_path",
    xlab = first_minus_other_path_label
  )
  plot_metric_by_difference(
    last_visited_difference_summary,
    y_col = "last_visited_path_chosen",
    ylab = "P(last visited path is chosen)",
    file_prefix = "revisit_last_visited_minus_mean_other_path_reward_vs_last_visited_path_chosen_by_sigma",
    x_col = "last_visited_minus_mean_other_path",
    xlab = last_minus_other_path_label
  )
  plot_metric_by_difference(
    node_difference_summary,
    y_col = "choose_first_observed_path",
    ylab = "P(choose first observed path)",
    file_prefix = "revisit_first_observed_minus_mean_other_path_reward_vs_probability_choose_first_observed_path_by_sigma",
    x_col = "first_observed_minus_mean_other_path",
    xlab = first_minus_other_path_label
  )
  plot_metric_by_difference(
    absolute_node_difference_summary,
    y_col = "timestep_before_stop",
    ylab = "Average timestep before stopping",
    file_prefix = "revisit_abs_first_observed_minus_mean_other_path_reward_vs_average_timestep_before_stop_by_sigma",
    x_col = "absolute_first_observed_minus_mean_other_path",
    xlab = abs_first_minus_other_path_label
  )
  plot_metric_by_difference(
    absolute_node_difference_summary,
    y_col = "kl_paid_total",
    ylab = "Average KL paid across timesteps",
    file_prefix = "revisit_abs_first_observed_minus_mean_other_path_reward_vs_average_kl_paid_by_sigma",
    x_col = "absolute_first_observed_minus_mean_other_path",
    xlab = abs_first_minus_other_path_label
  )
  plot_metric_by_difference(
    absolute_node_difference_summary,
    y_col = "kl_paid_per_stop_timestep",
    ylab = "Average KL paid\nper stop timestep",
    file_prefix = "revisit_abs_first_observed_minus_mean_other_path_reward_vs_average_kl_paid_per_timestep_before_stop_by_sigma",
    x_col = "absolute_first_observed_minus_mean_other_path",
    xlab = abs_first_minus_other_path_label
  )
  plot_metric_by_difference(
    absolute_node_difference_summary,
    y_col = "kl_paid_per_pre_stop_timestep",
    ylab = "Average KL paid /\n(observed rewards before stop - 1)",
    file_prefix = "revisit_abs_first_observed_minus_mean_other_path_reward_vs_average_kl_paid_per_pre_stop_timestep_by_sigma",
    x_col = "absolute_first_observed_minus_mean_other_path",
    xlab = abs_first_minus_other_path_label
  )
  plot_metric_by_difference(
    absolute_node_difference_summary,
    y_col = "kl_paid_at_stop_timestep",
    ylab = "Average KL paid\nat stopping timestep",
    file_prefix = "revisit_abs_first_observed_minus_mean_other_path_reward_vs_kl_paid_at_stop_timestep_by_sigma",
    x_col = "absolute_first_observed_minus_mean_other_path",
    xlab = abs_first_minus_other_path_label
  )
  plot_metric_by_difference(
    absolute_node_difference_summary,
    y_col = "kl_paid_at_stop_timestep_after_first_continue",
    ylab = "Last carried-forward KL\nbefore stopping\ncontinued after first reward",
    file_prefix = "revisit_abs_first_observed_minus_mean_other_path_reward_vs_kl_paid_at_stop_timestep_after_first_continue_by_sigma",
    x_col = "absolute_first_observed_minus_mean_other_path",
    xlab = abs_first_minus_other_path_label
  )
}
