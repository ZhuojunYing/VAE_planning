#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)

script_file_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
script_dir <- if (length(script_file_arg) > 0L) {
  dirname(normalizePath(sub("^--file=", "", script_file_arg[[1]]), mustWork = FALSE))
} else {
  "analyses/exp_binary"
}

usage <- function() {
  cat(
    "Usage:\n",
    "  Rscript analyses/exp_binary/plot_evidence_accumulation_beta_opp_comparison.R [evidence] [options]\n\n",
    "Loads trial-level CSVs written by model_jax/evidence_accumulation.py and writes one\n",
    "PNG per plot type comparing beta-vary and opportunity-vary evidence-accumulation behavior.\n\n",
    "Options:\n",
    "  --preset-file PATH          Preset CSV path. Default: analyses/exp_binary/evidence_accumulation_plot_presets.csv.\n",
    "  --input-dir DIR             Simulation CSV directory. Default: outputs/jax_simulations_evi.\n",
    "  --output-root DIR           Output root when --output-file is not supplied. Default: results.\n",
    "                              Default plots go under evidence_accumulation_compare/{observer_only,no_observer}.\n",
    "  --output-file PATH          Output PNG stem/path. Plot type suffixes are added.\n",
    "  --vary-memory-lambda-values LIST\n",
    "                              Memory-lambda values for the left column.\n",
    "                              Aliases: --memory-lambda-values, --memory-lambdas.\n",
    "  --vary-beta-values LIST     Legacy alias for --vary-memory-lambda-values.\n",
    "                              Aliases: --beta-values, --betas.\n",
    "  --vary-opportunity-values LIST\n",
    "                              Opportunity costs for the right column.\n",
    "                              Aliases: --opportunity-values, --opportunities, --opportunity-costs.\n",
    "  --fixed-opp VALUE           Opportunity cost held fixed for beta-vary curves.\n",
    "                              Alias: --fixed-opportunity, --fixed-opportunity-cost.\n",
    "  --fixed-memory-lambda VALUE Memory-lambda held fixed for opportunity-vary curves.\n",
    "  --fixed-beta VALUE          Legacy alias for --fixed-memory-lambda.\n",
    "  --observation-noise-std LIST\n",
    "                              Observation noise std value(s) to plot; multiple values add rows.\n",
    "                              Aliases: --obsstd, --sigma.\n",
    "                              If omitted, the most common value in the loaded files is used.\n",
    "  --loss-scale VALUE          Filter loss-scale value. Aliases: --lambda, --lambda-value.\n",
    "  --alpha VALUE               Filter alpha value. Default: no filter.\n",
    "  --seeds LIST                Filter seed values. Default: all available.\n",
    "  --rnn-units VALUE           Filter RNN units. Default: no filter.\n",
    "  --latent-dim VALUE          Filter latent dimension. Default: no filter.\n",
    "  --max-observations VALUE    Filter max observations. Alias: --maxobs. Default: no filter.\n",
    "  --correct-reward VALUE      Filter correct terminal reward scale. Default comes from preset, now 5.\n",
    "  --input-type VALUE          Filter trailing input type. Default: evidence.\n",
    "  --pay-kl-on-stop            Use CSVs with the _stop_paid filename suffix. Default comes from preset, now true.\n",
    "  --no-pay-kl-on-stop         Use legacy CSVs without the _stop_paid filename suffix.\n",
    "  --observer-only             Use CSVs with the _observer_endchoice filename suffix.\n",
    "  --no-observer-only          Use CSVs without the _observer_endchoice filename suffix.\n",
    "  --n-bins N                  Quantile bins for cumulative-evidence curve. Default: 20.\n",
    "  --min-samples N             Drop points with fewer than N trial samples. Default: 10.\n",
    "  --delta-min-samples N       Drop points in pooled/coherence-split delta plots with fewer\n",
    "                              than N trial samples. Default: same as --min-samples.\n",
    "  --delta-coherence-values LIST\n",
    "                              Also write coherence-specific delta and response-locked cumulative-evidence\n",
    "                              plots for these trial coherence\n",
    "                              magnitudes. Default: 0.05,0.2,0.4,0.8. Pass \"\" to disable.\n",
    "                              Timestep plots use policy_choose_a/b_t*, kl_d_t*, z_mu_*_t*,\n",
    "                              z_logvar_*_t*, prior_mu_*_t*, and prior_logvar_*_t* columns\n",
    "                              when available. z_mu/z_sigma analyses are normalized by the\n",
    "                              learned prior. Terminal logits are recovered as\n",
    "                              log(P_correct / P_incorrect).\n",
    "  --response-locked-coherence VALUE\n",
    "                              Coherence magnitude for response-locked plots. Alias: --fixed-coherence.\n",
    "                              If omitted, uses the most common nonzero coherence.\n",
    "  --max-steps-before-stop N   Keep response-locked timesteps -N,...,0. Default: 10.\n",
    "  --include-forced-stops      Include forced max-observation terminal decisions in response-locked plots.\n",
    "                              Default excludes them, matching plot_evidence_accumulation_response_locked.R.\n",
    "                              Non-observer runs also get a decision-threshold plot: choice-aligned\n",
    "                              cumulative evidence at stopping as a function of stopping duration.\n",
    "  --simple-fixed-obsstd VALUE  Also write a compact fixed-std folder with selected coherence values as rows.\n",
    "                              Aliases: --simple-obsstd, --fixed-std-simple, --fixed-obsstd-simple.\n",
    "  --simple-coherence-values LIST\n",
    "                              Coherence rows for --simple-fixed-obsstd plots. Default: same as\n",
    "                              --delta-coherence-values.\n",
    "  --simple-output-subdir NAME  Output subfolder suffix for simple plots. Default: observer_only_simple\n",
    "                              or no_observer_simple.\n",
    "  --help                      Show this message.\n\n",
    "Examples:\n",
    "  Rscript analyses/exp_binary/plot_evidence_accumulation_beta_opp_comparison.R evidence \\\n",
    "    --vary-beta-values \"1000,500,100\" \\\n",
    "    --vary-opportunity-values \"0.001,0.005,0.01\" \\\n",
    "    --observation-noise-std \"0.1,0.5,1.0\" \\\n",
    "    --correct-reward 5\n",
    sep = ""
  )
}

if (any(args %in% c("--help", "-h"))) {
  usage()
  quit(save = "no", status = 0L)
}

trim_string <- function(value) trimws(as.character(value))

extract_named_option <- function(args, option_names, default = NULL) {
  value <- default
  keep <- rep(TRUE, length(args))
  i <- 1L
  while (i <= length(args)) {
    arg <- args[[i]]
    matched <- NA_character_
    inline <- FALSE
    for (option_name in option_names) {
      if (identical(arg, option_name)) {
        matched <- option_name
        break
      }
      inline_prefix <- paste0(option_name, "=")
      if (startsWith(arg, inline_prefix)) {
        matched <- option_name
        inline <- TRUE
        break
      }
    }
    if (is.na(matched)) {
      i <- i + 1L
      next
    }
    if (inline) {
      value <- sub(paste0("^", matched, "="), "", arg)
      keep[[i]] <- FALSE
      i <- i + 1L
    } else {
      if (i == length(args)) {
        stop(sprintf("%s requires a value.", matched))
      }
      value <- args[[i + 1L]]
      keep[[i]] <- FALSE
      keep[[i + 1L]] <- FALSE
      i <- i + 2L
    }
  }
  list(args = args[keep], value = value)
}

extract_boolean_option <- function(args, true_names, false_names = character(), default = NULL) {
  value <- default
  keep <- rep(TRUE, length(args))
  truthy <- c("1", "true", "t", "yes", "y", "on")
  falsey <- c("0", "false", "f", "no", "n", "off")
  i <- 1L
  while (i <= length(args)) {
    arg <- args[[i]]
    matched <- NA_character_
    matched_value <- NA
    inline <- FALSE
    for (option_name in true_names) {
      if (identical(arg, option_name)) {
        matched <- option_name
        matched_value <- TRUE
        break
      }
      inline_prefix <- paste0(option_name, "=")
      if (startsWith(arg, inline_prefix)) {
        matched <- option_name
        inline <- TRUE
        break
      }
    }
    if (is.na(matched)) {
      for (option_name in false_names) {
        if (identical(arg, option_name)) {
          matched <- option_name
          matched_value <- FALSE
          break
        }
      }
    }
    if (is.na(matched)) {
      i <- i + 1L
      next
    }
    if (inline) {
      raw <- tolower(trim_string(sub(paste0("^", matched, "="), "", arg)))
      if (raw %in% truthy) {
        value <- TRUE
      } else if (raw %in% falsey) {
        value <- FALSE
      } else {
        stop(sprintf("%s expects true/false when using --flag=value syntax.", matched))
      }
      keep[[i]] <- FALSE
      i <- i + 1L
    } else {
      value <- matched_value
      keep[[i]] <- FALSE
      i <- i + 1L
    }
  }
  list(args = args[keep], value = value)
}

parse_bool_value <- function(value, default = FALSE, label = "boolean value") {
  if (is.null(value) || !nzchar(trim_string(value))) {
    return(default)
  }
  raw <- tolower(trim_string(value))
  if (raw %in% c("1", "true", "t", "yes", "y", "on")) {
    return(TRUE)
  }
  if (raw %in% c("0", "false", "f", "no", "n", "off")) {
    return(FALSE)
  }
  stop(sprintf("Could not parse %s as true/false: %s", label, value))
}

parse_csv_values <- function(value) {
  if (is.null(value) || !nzchar(trim_string(value))) {
    return(character())
  }
  out <- unlist(strsplit(as.character(value), "[,[:space:]]+"), use.names = FALSE)
  out <- trimws(out)
  out[nzchar(out)]
}

as_num <- function(value) suppressWarnings(as.numeric(as.character(value)))

num_label <- function(value) {
  value_num <- suppressWarnings(as.numeric(value))
  if (!is.finite(value_num)) {
    return(as.character(value))
  }
  format(signif(value_num, 6), scientific = FALSE, trim = TRUE)
}

values_label <- function(values) {
  values <- sort(unique(as_num(values)))
  values <- values[is.finite(values)]
  paste(vapply(values, num_label, character(1)), collapse = ",")
}

value_token <- function(value) {
  gsub("[^A-Za-z0-9]+", "p", num_label(value))
}

values_token <- function(values) {
  values <- sort(unique(as_num(values)))
  values <- values[is.finite(values)]
  paste(vapply(values, value_token, character(1)), collapse = "_")
}

parameter_equal <- function(x, y, tol = 1e-8) {
  x_num <- as_num(x)
  y_num <- as.numeric(y)
  is.finite(x_num) & is.finite(y_num) & abs(x_num - y_num) <= tol
}

snap_requested_values <- function(requested, available, label, tol = 1e-5) {
  requested <- sort(unique(as_num(requested)))
  requested <- requested[is.finite(requested)]
  available <- sort(unique(as_num(available)))
  available <- available[is.finite(available)]
  if (length(requested) == 0L || length(available) == 0L) {
    return(numeric())
  }
  out <- numeric()
  for (value in requested) {
    diffs <- abs(available - value)
    best_i <- which.min(diffs)
    if (length(best_i) == 0L || !is.finite(diffs[[best_i]]) || diffs[[best_i]] > tol) {
      warning(sprintf(
        "Requested %s=%s was not found within tolerance %g. Available: %s",
        label,
        num_label(value),
        tol,
        values_label(available)
      ))
      next
    }
    snapped <- available[[best_i]]
    if (abs(snapped - value) > 0) {
      message(sprintf(
        "Using stored %s=%s for requested %s=%s.",
        label,
        num_label(snapped),
        label,
        num_label(value)
      ))
    }
    out <- c(out, snapped)
  }
  sort(unique(out))
}

option_preset_file <- extract_named_option(
  args,
  c("--preset-file"),
  default = file.path(script_dir, "evidence_accumulation_plot_presets.csv")
)
args <- option_preset_file$args
option_input_dir <- extract_named_option(args, c("--input-dir"), default = NULL)
args <- option_input_dir$args
option_output_root <- extract_named_option(args, c("--output-root", "--results-dir"), default = NULL)
args <- option_output_root$args
option_output_file <- extract_named_option(args, c("--output-file", "--out"), default = NULL)
args <- option_output_file$args
option_memory_lambda_values <- extract_named_option(
  args,
  c("--vary-memory-lambda-values", "--memory-lambda-values", "--memory-lambdas"),
  default = NULL
)
args <- option_memory_lambda_values$args
option_beta_values <- extract_named_option(args, c("--vary-beta-values", "--beta-values", "--betas", "--vary-betas"), default = NULL)
args <- option_beta_values$args
option_opp_values <- extract_named_option(
  args,
  c("--vary-opportunity-values", "--opportunity-values", "--opportunities", "--opportunity-costs", "--vary-opps"),
  default = NULL
)
args <- option_opp_values$args
option_fixed_opp <- extract_named_option(
  args,
  c("--fixed-opp", "--fixed-opportunity", "--fixed-opportunity-cost"),
  default = NULL
)
args <- option_fixed_opp$args
option_fixed_memory_lambda <- extract_named_option(args, c("--fixed-memory-lambda", "--fixed-memory-lambda-value"), default = NULL)
args <- option_fixed_memory_lambda$args
option_fixed_beta <- extract_named_option(args, c("--fixed-beta"), default = NULL)
args <- option_fixed_beta$args
option_obsstd <- extract_named_option(args, c("--observation-noise-std", "--obsstd", "--sigma"), default = NULL)
args <- option_obsstd$args
option_loss_scale <- extract_named_option(args, c("--loss-scale", "--loss-scale-value", "--lambda", "--lambda-value"), default = NULL)
args <- option_loss_scale$args
option_alpha <- extract_named_option(args, c("--alpha"), default = NULL)
args <- option_alpha$args
option_seeds <- extract_named_option(args, c("--seeds"), default = NULL)
args <- option_seeds$args
option_rnn <- extract_named_option(args, c("--rnn-units", "--rnn-dims", "--rnn-dim"), default = NULL)
args <- option_rnn$args
option_latent <- extract_named_option(args, c("--latent-dim", "--latent-dims"), default = NULL)
args <- option_latent$args
option_maxobs <- extract_named_option(args, c("--max-observations", "--max-observations-before-stop", "--maxobs"), default = NULL)
args <- option_maxobs$args
option_correct_reward <- extract_named_option(args, c("--correct-reward", "--reward-scale", "--terminal-correct-reward"), default = NULL)
args <- option_correct_reward$args
option_input_type <- extract_named_option(args, c("--input-type"), default = NULL)
args <- option_input_type$args
option_pay_kl_on_stop <- extract_boolean_option(
  args,
  c("--pay-kl-on-stop", "--stop-paid"),
  c("--no-pay-kl-on-stop", "--no-stop-paid", "--legacy-no-stop-paid"),
  default = NULL
)
args <- option_pay_kl_on_stop$args
option_observer_only <- extract_boolean_option(
  args,
  c("--observer-only", "--choice-at-end-only", "--observer-endchoice", "--observer-end-choice"),
  c("--no-observer-only", "--no-choice-at-end-only", "--no-observer-endchoice", "--no-observer-end-choice"),
  default = NULL
)
args <- option_observer_only$args
option_n_bins <- extract_named_option(args, c("--n-bins", "--bins"), default = "20")
args <- option_n_bins$args
option_min_samples <- extract_named_option(args, c("--min-samples", "--min-n"), default = "10")
args <- option_min_samples$args
option_delta_min_samples <- extract_named_option(args, c("--delta-min-samples", "--min-delta-samples", "--delta-min-n"), default = NULL)
args <- option_delta_min_samples$args
option_delta_coherence_values <- extract_named_option(args, c("--delta-coherence-values", "--delta-coherences", "--coherence-values"), default = "0.05,0.2,0.4,0.8")
args <- option_delta_coherence_values$args
option_simple_fixed_obsstd <- extract_named_option(
  args,
  c("--simple-fixed-obsstd", "--simple-obsstd", "--fixed-std-simple", "--fixed-obsstd-simple"),
  default = NULL
)
args <- option_simple_fixed_obsstd$args
option_simple_coherence_values <- extract_named_option(args, c("--simple-coherence-values", "--simple-coherences"), default = NULL)
args <- option_simple_coherence_values$args
option_simple_output_subdir <- extract_named_option(args, c("--simple-output-subdir", "--simple-folder"), default = NULL)
args <- option_simple_output_subdir$args
option_response_locked_coherence <- extract_named_option(args, c("--response-locked-coherence", "--fixed-coherence"), default = NULL)
args <- option_response_locked_coherence$args
option_max_steps_before_stop <- extract_named_option(args, c("--max-steps-before-stop", "--max-relative-steps"), default = "10")
args <- option_max_steps_before_stop$args
option_include_forced_stops <- extract_boolean_option(
  args,
  c("--include-forced-stops", "--include-forced-terminal"),
  c("--exclude-forced-stops", "--exclude-forced-terminal"),
  default = FALSE
)
args <- option_include_forced_stops$args

if (length(args) > 1L) {
  usage()
  stop("Expected at most one positional label, e.g. evidence.")
}

preset_task <- if (length(args) == 1L) trim_string(args[[1L]]) else "evidence"

load_preset_rows <- function(preset_file, task) {
  if (!file.exists(preset_file)) {
    stop(sprintf("Preset file not found: %s", preset_file))
  }
  presets <- utils::read.csv(preset_file, stringsAsFactors = FALSE, check.names = FALSE)
  required <- c("task", "vary", "alpha_arg", "opportunity_arg", "input_dir", "results_dir")
  missing <- setdiff(required, names(presets))
  if (length(missing) > 0L) {
    stop(sprintf("Preset file %s is missing required column(s): %s", preset_file, paste(missing, collapse = ", ")))
  }
  if (!("memory_lambda_arg" %in% names(presets) || "beta_arg" %in% names(presets))) {
    stop(sprintf("Preset file %s must include memory_lambda_arg or legacy beta_arg.", preset_file))
  }
  rows <- presets[trimws(presets$task) == task, , drop = FALSE]
  vary_labels <- trimws(rows$vary)
  beta_row <- rows[vary_labels %in% c("memory_lambda", "memory", "beta"), , drop = FALSE]
  opp_row <- rows[trimws(rows$vary) == "opportunity", , drop = FALSE]
  if (nrow(beta_row) == 0L || nrow(opp_row) == 0L) {
    stop(sprintf("Need both memory_lambda and opportunity rows for task=%s in %s.", task, preset_file))
  }
  list(beta = beta_row[1L, , drop = FALSE], opportunity = opp_row[1L, , drop = FALSE])
}

preset_rows <- load_preset_rows(option_preset_file$value, preset_task)
preset_beta_row <- preset_rows$beta
preset_opp_row <- preset_rows$opportunity
preset_value <- function(row, column, default = NULL) {
  if (!column %in% names(row)) {
    return(default)
  }
  value <- row[[column]][[1L]]
  if (is.na(value) || !nzchar(trim_string(value))) {
    return(default)
  }
  trim_string(value)
}

if (is.null(option_input_dir$value)) option_input_dir$value <- preset_value(preset_beta_row, "input_dir", "outputs/jax_simulations_evi")
if (is.null(option_output_root$value)) option_output_root$value <- preset_value(preset_beta_row, "results_dir", "results")
if (is.null(option_input_type$value)) option_input_type$value <- preset_value(preset_beta_row, "input_type", "evidence")
if (is.null(option_memory_lambda_values$value)) {
  option_memory_lambda_values$value <- preset_value(preset_beta_row, "memory_lambda_arg", preset_value(preset_beta_row, "beta_arg", NULL))
}
if (is.null(option_opp_values$value)) option_opp_values$value <- preset_value(preset_opp_row, "opportunity_arg", NULL)
if (is.null(option_fixed_opp$value)) option_fixed_opp$value <- preset_value(preset_beta_row, "opportunity_arg", NULL)
if (is.null(option_fixed_memory_lambda$value)) {
  option_fixed_memory_lambda$value <- preset_value(preset_opp_row, "memory_lambda_arg", preset_value(preset_opp_row, "beta_arg", NULL))
}
if (is.null(option_obsstd$value)) option_obsstd$value <- preset_value(preset_beta_row, "observation_noise_std_arg", NULL)
if (is.null(option_loss_scale$value)) {
  option_loss_scale$value <- preset_value(preset_beta_row, "loss_scale_arg", preset_value(preset_beta_row, "lambda_arg", NULL))
}
if (is.null(option_alpha$value)) option_alpha$value <- preset_value(preset_beta_row, "alpha_arg", NULL)
user_seed_filter <- option_seeds$value
memory_seed_filter <- if (!is.null(user_seed_filter)) {
  user_seed_filter
} else {
  preset_value(preset_beta_row, "seed_arg", NULL)
}
opportunity_seed_filter <- if (!is.null(user_seed_filter)) {
  user_seed_filter
} else {
  preset_value(preset_opp_row, "seed_arg", memory_seed_filter)
}
memory_expansion_filter <- preset_value(preset_beta_row, "expansion_decision_version", NULL)
opportunity_expansion_filter <- preset_value(preset_opp_row, "expansion_decision_version", memory_expansion_filter)
memory_variant_filter <- preset_value(preset_beta_row, "model_variant", NULL)
opportunity_variant_filter <- preset_value(preset_opp_row, "model_variant", memory_variant_filter)
if (is.null(option_rnn$value)) option_rnn$value <- preset_value(preset_beta_row, "rnn_units_arg", NULL)
if (is.null(option_latent$value)) option_latent$value <- preset_value(preset_beta_row, "latent_dim_arg", NULL)
if (is.null(option_maxobs$value)) option_maxobs$value <- preset_value(preset_beta_row, "max_observations_arg", NULL)
if (is.null(option_correct_reward$value)) option_correct_reward$value <- preset_value(preset_beta_row, "correct_reward_arg", "5")
if (is.null(option_pay_kl_on_stop$value)) {
  option_pay_kl_on_stop$value <- parse_bool_value(
    preset_value(preset_beta_row, "pay_kl_on_stop_arg", "true"),
    default = TRUE,
    label = "pay_kl_on_stop_arg"
  )
}
if (is.null(option_observer_only$value)) {
  option_observer_only$value <- parse_bool_value(
    preset_value(preset_beta_row, "observer_only_arg", "false"),
    default = FALSE,
    label = "observer_only_arg"
  )
}

message(sprintf("Using evidence accumulation preset: task=%s from %s", preset_task, option_preset_file$value))

input_dir <- option_input_dir$value
output_root <- option_output_root$value
input_type <- trim_string(option_input_type$value)
n_bins <- as.integer(round(as.numeric(option_n_bins$value)))
minimum_samples <- as.integer(round(as.numeric(option_min_samples$value)))
delta_minimum_samples <- if (is.null(option_delta_min_samples$value) || !nzchar(trim_string(option_delta_min_samples$value))) {
  minimum_samples
} else {
  as.integer(round(as.numeric(option_delta_min_samples$value)))
}
delta_coherence_values <- as_num(parse_csv_values(option_delta_coherence_values$value))
delta_coherence_values <- sort(unique(delta_coherence_values[is.finite(delta_coherence_values)]))
simple_fixed_obsstd <- if (!is.null(option_simple_fixed_obsstd$value) && nzchar(trim_string(option_simple_fixed_obsstd$value))) {
  vals <- as_num(parse_csv_values(option_simple_fixed_obsstd$value))
  vals <- vals[is.finite(vals)]
  if (length(vals) != 1L) stop("--simple-fixed-obsstd expects exactly one numeric value.")
  vals[[1L]]
} else {
  NA_real_
}
simple_coherence_values <- if (!is.null(option_simple_coherence_values$value) && nzchar(trim_string(option_simple_coherence_values$value))) {
  vals <- as_num(parse_csv_values(option_simple_coherence_values$value))
  sort(unique(vals[is.finite(vals)]))
} else {
  delta_coherence_values
}
pay_kl_on_stop_mode <- isTRUE(option_pay_kl_on_stop$value)
observer_only_mode <- isTRUE(option_observer_only$value)
max_steps_before_stop <- as.integer(round(as.numeric(option_max_steps_before_stop$value)))
include_forced_stops <- isTRUE(option_include_forced_stops$value)
if (!is.finite(n_bins) || n_bins < 2L) stop("--n-bins must be at least 2.")
if (!is.finite(minimum_samples) || minimum_samples < 0L) stop("--min-samples must be nonnegative.")
if (!is.finite(delta_minimum_samples) || delta_minimum_samples < 0L) stop("--delta-min-samples must be nonnegative.")
if (!is.finite(max_steps_before_stop) || max_steps_before_stop < 0L) stop("--max-steps-before-stop must be a nonnegative integer.")
if (is.finite(simple_fixed_obsstd) && (is.null(option_obsstd$value) || !nzchar(trim_string(option_obsstd$value)))) {
  option_obsstd$value <- as.character(simple_fixed_obsstd)
}

read_csv_names <- function(path) {
  if (requireNamespace("data.table", quietly = TRUE)) {
    return(names(data.table::fread(path, nrows = 0L, showProgress = FALSE)))
  }
  names(utils::read.csv(path, nrows = 1L, check.names = FALSE))
}

read_csv_fast <- function(path, select = NULL) {
  if (requireNamespace("data.table", quietly = TRUE)) {
    return(as.data.frame(data.table::fread(path, select = select, showProgress = FALSE)))
  }
  dat <- utils::read.csv(path, check.names = FALSE)
  if (!is.null(select)) {
    select <- intersect(select, names(dat))
    dat <- dat[, select, drop = FALSE]
  }
  dat
}

rbind_fill <- function(frames) {
  frames <- frames[!vapply(frames, is.null, logical(1))]
  frames <- frames[vapply(frames, nrow, integer(1)) > 0L]
  if (length(frames) == 0L) {
    return(data.frame())
  }
  if (length(frames) == 1L) {
    return(as.data.frame(frames[[1L]], stringsAsFactors = FALSE))
  }
  if (requireNamespace("data.table", quietly = TRUE)) {
    return(as.data.frame(data.table::rbindlist(frames, fill = TRUE, use.names = TRUE)))
  }
  all_cols <- unique(unlist(lapply(frames, names), use.names = FALSE))
  frames <- lapply(frames, function(dat) {
    missing_cols <- setdiff(all_cols, names(dat))
    for (col in missing_cols) {
      dat[[col]] <- NA
    }
    dat[, all_cols, drop = FALSE]
  })
  do.call(rbind, unname(frames))
}

parse_evidence_filename_index <- function(input_dir) {
  files <- list.files(input_dir, pattern = "_evidence\\.csv$", recursive = TRUE, full.names = TRUE)
  files <- files[!grepl("_evidence_summary\\.csv$", basename(files))]
  if (length(files) == 0L) {
    return(data.frame())
  }
  basenames <- basename(files)
  common_tail <- paste0(
    "_expansion_([^_]+)_variant_([^_]+)_seed_([0-9]+)_coh_n([0-9]+)",
    "_min_([^_]+)_max_([^_]+)_obsstd_([^_]+)_maxobs_([0-9]+)",
    "_rnn_([^_]+)_latent_([^_]+)",
    "(?:_correctreward_([^_]+)(?:_incorrectreward_([^_]+))?)?",
    "(_stop_paid)?(_observer_endchoice)?_(.+)\\.csv$"
  )
  new_pattern <- paste0(
    "^evidence_loss_scale_([^_]+)_alpha_([^_]+)_beta_([^_]+)",
    "_memorylambda_([^_]+)_opportunity_([^_]+)",
    common_tail
  )
  old_pattern <- paste0(
    "^evidence_lambda_([^_]+)_alpha_([^_]+)_beta_([^_]+)_opportunity_([^_]+)",
    common_tail
  )

  parse_parts <- function(pattern, files_in, bases_in, legacy = FALSE) {
    matches <- regexec(pattern, bases_in, perl = TRUE)
    parts <- regmatches(bases_in, matches)
    keep <- lengths(parts) > 0L
    if (!any(keep)) return(data.frame())
    parts <- parts[keep]
    files_kept <- files_in[keep]
    part_at <- function(index) {
      vapply(parts, function(x) {
        if (length(x) >= index && !is.na(x[[index]])) x[[index]] else ""
      }, character(1))
    }
    if (legacy) {
      correct_reward <- as_num(part_at(16L))
      incorrect_reward <- as_num(part_at(17L))
      beta <- as_num(part_at(4L))
      loss_scale <- as_num(part_at(2L))
      memory_lambda <- 1 / beta
      out <- data.frame(
        file = files_kept,
        loss_scale = loss_scale,
        lambda = loss_scale,
        memory_lambda = memory_lambda,
        alpha = as_num(part_at(3L)),
        beta = beta,
        opportunity = as_num(part_at(5L)),
        expansion = part_at(6L),
        variant = part_at(7L),
        seed = as.integer(part_at(8L)),
        coherence_n = as.integer(part_at(9L)),
        coherence_min = as_num(part_at(10L)),
        coherence_max = as_num(part_at(11L)),
        observation_noise_std = as_num(part_at(12L)),
        max_observations = as.integer(part_at(13L)),
        rnn_units = as.integer(part_at(14L)),
        latent_dim = as.integer(part_at(15L)),
        correct_reward = correct_reward,
        incorrect_reward = incorrect_reward,
        pay_kl_on_stop = nzchar(part_at(18L)),
        choice_at_end_only = nzchar(part_at(19L)),
        input_type = part_at(20L),
        filename_schema = "legacy_lambda",
        stringsAsFactors = FALSE
      )
    } else {
      correct_reward <- as_num(part_at(17L))
      incorrect_reward <- as_num(part_at(18L))
      loss_scale <- as_num(part_at(2L))
      out <- data.frame(
        file = files_kept,
        loss_scale = loss_scale,
        lambda = loss_scale,
        memory_lambda = as_num(part_at(5L)),
        alpha = as_num(part_at(3L)),
        beta = as_num(part_at(4L)),
        opportunity = as_num(part_at(6L)),
        expansion = part_at(7L),
        variant = part_at(8L),
        seed = as.integer(part_at(9L)),
        coherence_n = as.integer(part_at(10L)),
        coherence_min = as_num(part_at(11L)),
        coherence_max = as_num(part_at(12L)),
        observation_noise_std = as_num(part_at(13L)),
        max_observations = as.integer(part_at(14L)),
        rnn_units = as.integer(part_at(15L)),
        latent_dim = as.integer(part_at(16L)),
        correct_reward = correct_reward,
        incorrect_reward = incorrect_reward,
        pay_kl_on_stop = nzchar(part_at(19L)),
        choice_at_end_only = nzchar(part_at(20L)),
        input_type = part_at(21L),
        filename_schema = "loss_scale_memorylambda",
        stringsAsFactors = FALSE
      )
    }
    out$correct_reward[!is.finite(out$correct_reward)] <- 1.0
    out$incorrect_reward[!is.finite(out$incorrect_reward)] <- 0.0
    out
  }

  parsed_list <- list(
    parse_parts(new_pattern, files, basenames, legacy = FALSE),
    parse_parts(old_pattern, files, basenames, legacy = TRUE)
  )
  parsed_list <- parsed_list[vapply(parsed_list, nrow, integer(1)) > 0L]
  if (length(parsed_list) == 0L) {
    return(data.frame())
  }
  rbind_fill(parsed_list)
}

filter_numeric_option <- function(index, column, value, label, tol = 1e-8) {
  if (is.null(value) || !nzchar(trim_string(value))) {
    return(index)
  }
  values <- as_num(parse_csv_values(value))
  values <- values[is.finite(values)]
  if (length(values) == 0L) {
    stop(sprintf("%s must contain numeric value(s).", label))
  }
  keep <- rep(FALSE, nrow(index))
  for (v in values) {
    keep <- keep | parameter_equal(index[[column]], v, tol = tol)
  }
  index[keep, , drop = FALSE]
}

numeric_option_mask <- function(index, column, value, label, tol = 1e-8) {
  if (is.null(value) || !nzchar(trim_string(value))) {
    return(rep(TRUE, nrow(index)))
  }
  values <- as_num(parse_csv_values(value))
  values <- values[is.finite(values)]
  if (length(values) == 0L) {
    stop(sprintf("%s must contain numeric value(s).", label))
  }
  keep <- rep(FALSE, nrow(index))
  for (value_i in values) {
    keep <- keep | parameter_equal(index[[column]], value_i, tol = tol)
  }
  keep
}

string_option_mask <- function(index, column, value, label) {
  if (is.null(value) || !nzchar(trim_string(value))) {
    return(rep(TRUE, nrow(index)))
  }
  values <- tolower(trimws(parse_csv_values(value)))
  values <- values[nzchar(values)]
  if (length(values) == 0L) {
    return(rep(TRUE, nrow(index)))
  }
  if (!column %in% names(index)) {
    stop(sprintf("Cannot filter %s because metadata column %s is missing.", label, column))
  }
  tolower(trimws(as.character(index[[column]]))) %in% values
}

most_common_numeric <- function(values) {
  values <- as_num(values)
  values <- values[is.finite(values)]
  if (length(values) == 0L) return(NA_real_)
  tab <- sort(table(format(values, digits = 16)), decreasing = TRUE)
  as.numeric(names(tab)[[1L]])
}

select_fixed_value <- function(values, requested, preferred, label) {
  values <- sort(unique(as_num(values)))
  values <- values[is.finite(values)]
  if (length(values) == 0L) {
    stop(sprintf("No available %s values.", label))
  }
  if (!is.null(requested) && nzchar(trim_string(requested))) {
    value <- as_num(parse_csv_values(requested)[[1L]])
    if (!any(parameter_equal(values, value))) {
      stop(sprintf("Requested fixed %s=%s was not found. Available: %s", label, num_label(value), values_label(values)))
    }
    return(value)
  }
  if (is.finite(preferred) && any(parameter_equal(values, preferred))) {
    message(sprintf("Selected fixed %s=%s using the reference-script default.", label, num_label(preferred)))
    return(preferred)
  }
  value <- most_common_numeric(values)
  message(sprintf(
    "Reference default fixed %s=%s was unavailable; selected most common %s=%s.",
    label,
    num_label(preferred),
    label,
    num_label(value)
  ))
  value
}

coerce_logical <- function(x) {
  if (is.logical(x)) return(x)
  if (is.numeric(x)) return(x != 0)
  key <- tolower(trimws(as.character(x)))
  key %in% c("true", "t", "1", "yes", "y")
}

step_columns <- function(cols, prefix) {
  out <- grep(paste0("^", prefix, "[0-9]+$"), cols, value = TRUE)
  steps <- as.integer(sub(prefix, "", out))
  out[order(steps)]
}

load_one_evidence_file <- function(path, meta) {
  columns <- read_csv_names(path)
  required_candidates <- c(
    "graph", "correct_choice", "correct_action", "coherence", "signed_coherence",
    "num_observations", "stopping_time", "terminal_action", "choose_right",
    "choose_correct", "correct", "total_kl_paid", "decision_cumulative_evidence",
    "cumulative_evidence_at_decision"
  )
  if (!any(c("decision_cumulative_evidence", "cumulative_evidence_at_decision") %in% columns)) {
    required_candidates <- c(required_candidates, grep("^cumulative_evidence_t[0-9]+$", columns, value = TRUE))
  }
  timestep_columns <- grep(
    "^(evidence_sample|cumulative_evidence|policy_continue|policy_choose_a|policy_choose_b|full_policy_continue|full_policy_choose_a|full_policy_choose_b|kl_d|action|stop)_t[0-9]+$|^(z_mu|z_logvar|z_sigma|prior_mu|prior_logvar|prior_sigma)_[0-9]+_t[0-9]+$",
    columns,
    value = TRUE
  )
  required_candidates <- c(required_candidates, timestep_columns)
  selected <- intersect(required_candidates, columns)
  dat <- read_csv_fast(path, select = selected)
  if (!"coherence" %in% names(dat)) {
    stop(sprintf("Missing required column coherence in %s.", path))
  }
  if (!"terminal_action" %in% names(dat)) {
    stop(sprintf("Missing required column terminal_action in %s.", path))
  }
  if (!"signed_coherence" %in% names(dat)) {
    if (!all(c("correct_choice", "coherence") %in% names(dat))) {
      stop(sprintf("Missing signed_coherence and cannot derive it in %s.", path))
    }
    dat$signed_coherence <- as_num(dat$correct_choice) * as_num(dat$coherence)
  }
  if (!"choose_right" %in% names(dat)) {
    dat$choose_right <- as_num(dat$terminal_action) == 2
  } else {
    dat$choose_right <- coerce_logical(dat$choose_right)
  }
  if (!"choose_correct" %in% names(dat)) {
    if ("correct" %in% names(dat)) {
      dat$choose_correct <- coerce_logical(dat$correct)
    } else if ("correct_action" %in% names(dat)) {
      dat$choose_correct <- as_num(dat$terminal_action) == as_num(dat$correct_action)
    } else {
      stop(sprintf("Missing choose_correct/correct and cannot derive it in %s.", path))
    }
  } else {
    dat$choose_correct <- coerce_logical(dat$choose_correct)
  }
  if (!"num_observations" %in% names(dat)) {
    if ("stopping_time" %in% names(dat)) {
      dat$num_observations <- as_num(dat$stopping_time)
    } else {
      stop(sprintf("Missing num_observations/stopping_time in %s.", path))
    }
  }
  if (!"stopping_time" %in% names(dat)) {
    dat$stopping_time <- dat$num_observations
  }
  if (!"decision_cumulative_evidence" %in% names(dat)) {
    if ("cumulative_evidence_at_decision" %in% names(dat)) {
      dat$decision_cumulative_evidence <- as_num(dat$cumulative_evidence_at_decision)
    } else {
      cumulative_cols <- grep("^cumulative_evidence_t[0-9]+$", names(dat), value = TRUE)
      if (length(cumulative_cols) == 0L) {
        stop(sprintf("No cumulative evidence column found in %s.", path))
      }
      cumulative_steps <- as.integer(sub("^cumulative_evidence_t", "", cumulative_cols))
      ordered_cols <- cumulative_cols[order(cumulative_steps)]
      mat <- as.matrix(dat[, ordered_cols, drop = FALSE])
      obs <- pmin(pmax(as.integer(round(as_num(dat$num_observations))), 1L), ncol(mat))
      dat$decision_cumulative_evidence <- mat[cbind(seq_len(nrow(mat)), obs)]
    }
  }
  dat$choose_right <- as.numeric(dat$choose_right)
  dat$choose_correct <- as.numeric(dat$choose_correct)
  dat$coherence <- as_num(dat$coherence)
  dat$signed_coherence <- as_num(dat$signed_coherence)
  dat$num_observations <- as_num(dat$num_observations)
  dat$stopping_time <- as_num(dat$stopping_time)
  dat$decision_cumulative_evidence <- as_num(dat$decision_cumulative_evidence)
  if ("total_kl_paid" %in% names(dat)) {
    dat$total_kl_paid <- as_num(dat$total_kl_paid)
  } else {
    kl_cols <- grep("^kl_d_t[0-9]+$", names(dat), value = TRUE)
    if (length(kl_cols) > 0L) {
      kl_steps <- as.integer(sub("^kl_d_t", "", kl_cols))
      kl_cols <- kl_cols[order(kl_steps)]
      kl_mat <- as.matrix(data.frame(lapply(kl_cols, function(col) as_num(dat[[col]])), check.names = FALSE))
      stop_steps <- pmin(pmax(as.integer(round(as_num(dat$num_observations))), 1L), ncol(kl_mat))
      step_mat <- matrix(
        rep(seq_len(ncol(kl_mat)), each = nrow(kl_mat)),
        nrow = nrow(kl_mat),
        ncol = ncol(kl_mat)
      )
      kl_mat[!is.finite(stop_steps) | step_mat > stop_steps] <- NA_real_
      dat$total_kl_paid <- rowSums(kl_mat, na.rm = TRUE)
    } else {
      dat$total_kl_paid <- NA_real_
    }
  }
  dat$beta <- meta$beta[[1L]]
  dat$opportunity <- meta$opportunity[[1L]]
  dat$seed <- meta$seed[[1L]]
  dat$run_id <- basename(path)
  dat$loss_scale <- meta$loss_scale[[1L]]
  dat$memory_lambda <- meta$memory_lambda[[1L]]
  dat$lambda <- meta$loss_scale[[1L]]
  dat$alpha <- meta$alpha[[1L]]
  dat$observation_noise_std <- meta$observation_noise_std[[1L]]
  dat$max_observations <- meta$max_observations[[1L]]
  dat$pay_kl_on_stop <- isTRUE(meta$pay_kl_on_stop[[1L]])
  dat$choice_at_end_only <- isTRUE(meta$choice_at_end_only[[1L]])
  dat$correct_reward <- meta$correct_reward[[1L]]
  dat$incorrect_reward <- meta$incorrect_reward[[1L]]
  dat$rnn_units <- meta$rnn_units[[1L]]
  dat$latent_dim <- meta$latent_dim[[1L]]
  dat$file <- path
  dat
}

mean_or_na <- function(x) {
  x <- as_num(x)
  x <- x[is.finite(x)]
  if (length(x) == 0L) NA_real_ else mean(x)
}

row_mean_or_na <- function(mat) {
  if (is.null(mat) || length(mat) == 0L) {
    return(numeric())
  }
  if (is.null(dim(mat))) {
    mat <- matrix(mat, ncol = 1L)
  }
  out <- rowMeans(mat, na.rm = TRUE)
  out[!is.finite(out)] <- NA_real_
  out
}

sem_or_na <- function(x) {
  x <- as_num(x)
  x <- x[is.finite(x)]
  if (length(x) < 2L) NA_real_ else stats::sd(x) / sqrt(length(x))
}

summarize_across_runs <- function(dat, group_cols, value_col, x_cols = character(), min_samples = minimum_samples) {
  if (nrow(dat) == 0L) return(data.frame())
  keep <- is.finite(as_num(dat[[value_col]]))
  dat <- dat[keep, , drop = FALSE]
  if (nrow(dat) == 0L) return(data.frame())
  run_group_cols <- unique(c(group_cols, "run_id"))
  if (requireNamespace("data.table", quietly = TRUE)) {
    dt <- data.table::as.data.table(dat)
    run_summary <- dt[, c(
      list(value = mean_or_na(get(value_col)), n_trials = .N),
      stats::setNames(lapply(x_cols, function(col) mean_or_na(get(col))), x_cols)
    ), by = run_group_cols]
    out <- run_summary[, c(
      list(
        value = mean_or_na(value),
        value_sem = sem_or_na(value),
        n_runs = data.table::uniqueN(run_id),
        n_trials = sum(n_trials)
      ),
      stats::setNames(lapply(x_cols, function(col) mean_or_na(get(col))), x_cols)
    ), by = group_cols]
    out <- as.data.frame(out)
  } else {
    run_summary <- aggregate(dat[[value_col]], by = dat[, run_group_cols, drop = FALSE], FUN = mean_or_na)
    names(run_summary)[names(run_summary) == "x"] <- "value"
    counts <- aggregate(rep(1, nrow(dat)), by = dat[, run_group_cols, drop = FALSE], FUN = sum)
    names(counts)[names(counts) == "x"] <- "n_trials"
    run_summary <- merge(run_summary, counts, by = run_group_cols, all.x = TRUE)
    for (x_col in x_cols) {
      x_sum <- aggregate(dat[[x_col]], by = dat[, run_group_cols, drop = FALSE], FUN = mean_or_na)
      names(x_sum)[names(x_sum) == "x"] <- x_col
      run_summary <- merge(run_summary, x_sum, by = run_group_cols, all.x = TRUE)
    }
    out <- aggregate(run_summary$value, by = run_summary[, group_cols, drop = FALSE], FUN = mean_or_na)
    names(out)[names(out) == "x"] <- "value"
    sems <- aggregate(run_summary$value, by = run_summary[, group_cols, drop = FALSE], FUN = sem_or_na)
    names(sems)[names(sems) == "x"] <- "value_sem"
    out <- merge(out, sems, by = group_cols, all.x = TRUE)
    runs <- aggregate(run_summary$run_id, by = run_summary[, group_cols, drop = FALSE], FUN = function(x) length(unique(x)))
    names(runs)[names(runs) == "x"] <- "n_runs"
    out <- merge(out, runs, by = group_cols, all.x = TRUE)
    trial_counts <- aggregate(run_summary$n_trials, by = run_summary[, group_cols, drop = FALSE], FUN = sum)
    names(trial_counts)[names(trial_counts) == "x"] <- "n_trials"
    out <- merge(out, trial_counts, by = group_cols, all.x = TRUE)
    for (x_col in x_cols) {
      x_sum <- aggregate(run_summary[[x_col]], by = run_summary[, group_cols, drop = FALSE], FUN = mean_or_na)
      names(x_sum)[names(x_sum) == "x"] <- x_col
      out <- merge(out, x_sum, by = group_cols, all.x = TRUE)
    }
  }
  out <- out[out$n_trials >= min_samples, , drop = FALSE]
  names(out)[names(out) == "value"] <- value_col
  names(out)[names(out) == "value_sem"] <- paste0(value_col, "_sem")
  out
}

empty_timestep_data <- function() {
  data.frame(
    family = character(),
    parameter_value = numeric(),
    observation_noise_std = numeric(),
    run_id = character(),
    coherence = numeric(),
    timestep_before_stopping = numeric(),
    terminal_choice_logit_correct = numeric(),
    kl_paid = numeric(),
    abs_delta_action_aligned_action_logit = numeric(),
    abs_delta_z_mu = numeric(),
    signed_delta_z_sigma = numeric(),
    average_z_sigma = numeric(),
    stringsAsFactors = FALSE
  )
}

latent_step_matrix <- function(dat, prefix, step, transform = NULL) {
  cols <- grep(sprintf("^%s_[0-9]+_t%d$", prefix, step), names(dat), value = TRUE)
  if (length(cols) == 0L) {
    return(NULL)
  }
  dim_ids <- as.integer(sub(sprintf("^%s_([0-9]+)_t%d$", prefix, step), "\\1", cols))
  cols <- cols[order(dim_ids)]
  mat <- as.matrix(data.frame(lapply(cols, function(col) as_num(dat[[col]])), check.names = FALSE))
  if (!is.null(transform)) {
    mat <- transform(mat)
  }
  mat
}

clip_exp_half <- function(mat) {
  exp(0.5 * pmax(pmin(mat, 30), -30))
}

same_latent_shape <- function(...) {
  mats <- list(...)
  mats <- mats[!vapply(mats, is.null, logical(1))]
  if (length(mats) < 2L) {
    return(FALSE)
  }
  ncols <- vapply(mats, ncol, integer(1))
  all(ncols == ncols[[1L]])
}

build_timestep_data <- function(dat) {
  policy_a_cols <- grep("^policy_choose_a_t[0-9]+$", names(dat), value = TRUE)
  policy_b_cols <- grep("^policy_choose_b_t[0-9]+$", names(dat), value = TRUE)
  kl_cols <- grep("^kl_d_t[0-9]+$", names(dat), value = TRUE)
  z_mu_cols <- grep("^z_mu_[0-9]+_t[0-9]+$", names(dat), value = TRUE)
  z_logvar_cols <- grep("^z_logvar_[0-9]+_t[0-9]+$", names(dat), value = TRUE)
  z_sigma_cols <- grep("^z_sigma_[0-9]+_t[0-9]+$", names(dat), value = TRUE)
  prior_mu_cols <- grep("^prior_mu_[0-9]+_t[0-9]+$", names(dat), value = TRUE)
  prior_logvar_cols <- grep("^prior_logvar_[0-9]+_t[0-9]+$", names(dat), value = TRUE)
  prior_sigma_cols <- grep("^prior_sigma_[0-9]+_t[0-9]+$", names(dat), value = TRUE)
  steps <- sort(unique(as.integer(sub(".*_t", "", c(
    policy_a_cols, policy_b_cols, kl_cols, z_mu_cols, z_logvar_cols, z_sigma_cols,
    prior_mu_cols, prior_logvar_cols, prior_sigma_cols
  )))))
  steps <- steps[is.finite(steps)]
  if (length(steps) == 0L) {
    return(empty_timestep_data())
  }
  pieces <- vector("list", length(steps))
  piece_i <- 0L
  prev_action_aligned_logit <- rep(NA_real_, nrow(dat))
  prev_z_mu <- NULL
  prev_z_sigma <- NULL
  warned_missing_prior <- FALSE
  max_step <- max(steps)
  # Match plot_evidence_accumulation_response_locked.R:
  # relative_timestep = step - num_observations, so tau = 0 is the stop-aligned
  # terminal policy decision after the final observed evidence sample.
  stop_time_saved <- as.integer(round(as_num(dat$num_observations)))
  stop_time_saved[!is.finite(stop_time_saved)] <- NA_integer_
  stop_time_saved <- pmin(stop_time_saved, max_step)
  for (step in steps) {
    a_col <- sprintf("policy_choose_a_t%d", step)
    b_col <- sprintf("policy_choose_b_t%d", step)
    kl_col <- sprintf("kl_d_t%d", step)
    if (!all(c(a_col, b_col) %in% names(dat))) {
      next
    }
    relative_timestep_all <- step - stop_time_saved
    keep <- is.finite(stop_time_saved) & stop_time_saved >= step
    if (!any(keep)) {
      next
    }
    p_a_all <- as_num(dat[[a_col]])
    p_b_all <- as_num(dat[[b_col]])
    signed_coherence_all <- as_num(dat$signed_coherence)
    p_correct_all <- ifelse(signed_coherence_all > 0, p_b_all, ifelse(signed_coherence_all < 0, p_a_all, NA_real_))
    p_incorrect_all <- ifelse(signed_coherence_all > 0, p_a_all, ifelse(signed_coherence_all < 0, p_b_all, NA_real_))
    terminal_denom_all <- p_a_all + p_b_all
    terminal_choice_logit_correct_all <- ifelse(
      is.finite(p_correct_all) & is.finite(p_incorrect_all) & p_correct_all >= 0 & p_incorrect_all >= 0 & terminal_denom_all > 0,
      log(pmax(p_correct_all, 1e-12)) - log(pmax(p_incorrect_all, 1e-12)),
      NA_real_
    )
    terminal_action_all <- as.integer(round(as_num(dat$terminal_action)))
    p_action_aligned_all <- ifelse(
      terminal_action_all == 2L,
      p_b_all,
      ifelse(terminal_action_all == 1L, p_a_all, NA_real_)
    )
    p_action_opposed_all <- ifelse(
      terminal_action_all == 2L,
      p_a_all,
      ifelse(terminal_action_all == 1L, p_b_all, NA_real_)
    )
    action_aligned_logit_all <- ifelse(
      is.finite(p_action_aligned_all) & is.finite(p_action_opposed_all) &
        p_action_aligned_all >= 0 & p_action_opposed_all >= 0 & terminal_denom_all > 0,
      log(pmax(p_action_aligned_all, 1e-12)) - log(pmax(p_action_opposed_all, 1e-12)),
      NA_real_
    )
    abs_delta_action_aligned_logit_all <- abs(action_aligned_logit_all - prev_action_aligned_logit)

    z_mu_raw <- latent_step_matrix(dat, "z_mu", step)
    prior_mu_now <- latent_step_matrix(dat, "prior_mu", step)
    prior_sigma_now <- latent_step_matrix(dat, "prior_sigma", step)
    if (is.null(prior_sigma_now)) {
      prior_sigma_now <- latent_step_matrix(dat, "prior_logvar", step, transform = clip_exp_half)
    }
    if (!is.null(z_mu_raw) && same_latent_shape(z_mu_raw, prior_mu_now, prior_sigma_now)) {
      z_mu_now <- (z_mu_raw - prior_mu_now) / pmax(prior_sigma_now, 1e-8)
    } else {
      z_mu_now <- NULL
      if (!is.null(z_mu_raw) && !warned_missing_prior) {
        warning("Prior columns are missing or mismatched; skipping prior-normalized z_mu/z_sigma timestep analyses.")
        warned_missing_prior <- TRUE
      }
    }
    if (!is.null(z_mu_now) && !is.null(prev_z_mu) && ncol(z_mu_now) == ncol(prev_z_mu)) {
      # Collapse vector latents to a per-trial scalar as mean absolute dimension-wise change.
      abs_delta_z_mu_all <- row_mean_or_na(abs(z_mu_now - prev_z_mu))
    } else {
      abs_delta_z_mu_all <- rep(NA_real_, nrow(dat))
    }

    z_sigma_raw <- latent_step_matrix(dat, "z_sigma", step)
    if (!is.null(z_sigma_raw)) {
      z_sigma_now <- z_sigma_raw
    } else {
      z_sigma_now <- latent_step_matrix(dat, "z_logvar", step, transform = clip_exp_half)
    }
    if (!is.null(z_sigma_now) && same_latent_shape(z_sigma_now, prior_sigma_now)) {
      z_sigma_now <- z_sigma_now / pmax(prior_sigma_now, 1e-8)
    } else if (!is.null(z_sigma_now)) {
      z_sigma_now <- NULL
      if (!warned_missing_prior) {
        warning("Prior columns are missing or mismatched; skipping prior-normalized z_mu/z_sigma timestep analyses.")
        warned_missing_prior <- TRUE
      }
    }
    if (!is.null(z_sigma_now) && !is.null(prev_z_sigma) && ncol(z_sigma_now) == ncol(prev_z_sigma)) {
      signed_delta_z_sigma_all <- row_mean_or_na(z_sigma_now - prev_z_sigma)
    } else {
      signed_delta_z_sigma_all <- rep(NA_real_, nrow(dat))
    }
    average_z_sigma_all <- if (!is.null(z_sigma_now)) {
      row_mean_or_na(z_sigma_now)
    } else {
      rep(NA_real_, nrow(dat))
    }

    piece_i <- piece_i + 1L
    pieces[[piece_i]] <- data.frame(
      family = dat$family[keep],
      parameter_value = as_num(dat$parameter_value[keep]),
      observation_noise_std = as_num(dat$observation_noise_std[keep]),
      run_id = dat$run_id[keep],
      coherence = as_num(dat$coherence[keep]),
      timestep_before_stopping = relative_timestep_all[keep],
      terminal_choice_logit_correct = terminal_choice_logit_correct_all[keep],
      kl_paid = if (kl_col %in% names(dat)) as_num(dat[[kl_col]][keep]) else NA_real_,
      abs_delta_action_aligned_action_logit = abs_delta_action_aligned_logit_all[keep],
      abs_delta_z_mu = abs_delta_z_mu_all[keep],
      signed_delta_z_sigma = signed_delta_z_sigma_all[keep],
      average_z_sigma = average_z_sigma_all[keep],
      stringsAsFactors = FALSE
    )
    prev_action_aligned_logit <- action_aligned_logit_all
    if (!is.null(z_mu_now)) {
      prev_z_mu <- z_mu_now
    }
    if (!is.null(z_sigma_now)) {
      prev_z_sigma <- z_sigma_now
    }
  }
  if (piece_i == 0L) {
    return(empty_timestep_data())
  }
  rbind_fill(pieces[seq_len(piece_i)])
}

empty_response_locked_data <- function() {
  data.frame(
    family = character(),
    parameter_value = numeric(),
    observation_noise_std = numeric(),
    run_id = character(),
    trial_id = character(),
    coherence = numeric(),
    timestep = integer(),
    stopping_timestep = integer(),
    relative_timestep = integer(),
    choice_aligned_cumulative_evidence = numeric(),
    p_stop = numeric(),
    p_eventual_choice_given_stop = numeric(),
    forced_terminal = logical(),
    stringsAsFactors = FALSE
  )
}

build_response_locked_data <- function(dat, max_steps_before_stop = 10L) {
  cols <- names(dat)
  cumulative_cols <- step_columns(cols, "cumulative_evidence_t")
  p_continue_cols <- step_columns(cols, "policy_continue_t")
  p_a_cols <- step_columns(cols, "policy_choose_a_t")
  p_b_cols <- step_columns(cols, "policy_choose_b_t")
  evidence_cols <- step_columns(cols, "evidence_sample_t")
  action_cols <- step_columns(cols, "action_t")
  stop_cols <- step_columns(cols, "stop_t")
  step_ids <- sort(unique(as.integer(sub(".*_t", "", c(
    cumulative_cols, p_continue_cols, p_a_cols, p_b_cols, evidence_cols, action_cols, stop_cols
  )))))
  step_ids <- step_ids[is.finite(step_ids)]
  if (length(step_ids) == 0L || length(p_a_cols) == 0L || length(p_b_cols) == 0L || length(cumulative_cols) == 0L) {
    return(empty_response_locked_data())
  }
  trial_id <- if ("graph" %in% names(dat)) {
    paste(dat$run_id, dat$graph, sep = "::")
  } else {
    paste(dat$run_id, seq_len(nrow(dat)), sep = "::")
  }
  pieces <- vector("list", length(step_ids))
  piece_i <- 0L
  stop_time_saved <- as.integer(round(as_num(dat$num_observations)))
  stop_time_saved[!is.finite(stop_time_saved)] <- NA_integer_
  terminal_action <- as.integer(round(as_num(dat$terminal_action)))
  max_observations <- as.integer(round(as_num(dat$max_observations)))
  forced_terminal_all <- is.finite(stop_time_saved) & is.finite(max_observations) & stop_time_saved >= max_observations
  for (step in step_ids) {
    a_col <- sprintf("policy_choose_a_t%d", step)
    b_col <- sprintf("policy_choose_b_t%d", step)
    cumulative_col <- sprintf("cumulative_evidence_t%d", step)
    if (!all(c(a_col, b_col, cumulative_col) %in% names(dat))) {
      next
    }
    valid <- is.finite(stop_time_saved) & stop_time_saved >= step & terminal_action %in% c(1L, 2L)
    relative_timestep <- step - stop_time_saved
    keep <- valid & relative_timestep >= -max_steps_before_stop & relative_timestep <= 0
    if (!any(keep)) {
      next
    }
    p_a <- as_num(dat[[a_col]])
    p_b <- as_num(dat[[b_col]])
    p_stop <- p_a + p_b
    p_eventual_choice <- ifelse(terminal_action == 1L, p_a, p_b)
    p_eventual_choice_given_stop <- p_eventual_choice / pmax(p_stop, 1e-12)
    cumulative_evidence <- as_num(dat[[cumulative_col]])
    choice_direction <- ifelse(terminal_action == 2L, 1, -1)
    piece_i <- piece_i + 1L
    pieces[[piece_i]] <- data.frame(
      family = dat$family[keep],
      parameter_value = as_num(dat$parameter_value[keep]),
      observation_noise_std = as_num(dat$observation_noise_std[keep]),
      run_id = dat$run_id[keep],
      trial_id = trial_id[keep],
      coherence = as_num(dat$coherence[keep]),
      timestep = step,
      stopping_timestep = stop_time_saved[keep],
      relative_timestep = relative_timestep[keep],
      choice_aligned_cumulative_evidence = choice_direction[keep] * cumulative_evidence[keep],
      p_stop = p_stop[keep],
      p_eventual_choice_given_stop = p_eventual_choice_given_stop[keep],
      forced_terminal = forced_terminal_all[keep],
      stringsAsFactors = FALSE
    )
  }
  if (piece_i == 0L) {
    return(empty_response_locked_data())
  }
  rbind_fill(pieces[seq_len(piece_i)])
}

build_full_policy_target_response_data <- function(dat, max_steps_before_stop = 10L) {
  cols <- names(dat)
  full_a_cols <- step_columns(cols, "full_policy_choose_a_t")
  full_b_cols <- step_columns(cols, "full_policy_choose_b_t")
  if (length(full_a_cols) == 0L || length(full_b_cols) == 0L) {
    return(data.frame())
  }
  step_ids <- sort(unique(as.integer(sub(".*_t", "", c(full_a_cols, full_b_cols)))))
  step_ids <- step_ids[is.finite(step_ids)]
  if (length(step_ids) == 0L) {
    return(data.frame())
  }
  trial_id <- if ("graph" %in% names(dat)) {
    paste(dat$run_id, dat$graph, sep = "::")
  } else {
    paste(dat$run_id, seq_len(nrow(dat)), sep = "::")
  }
  stop_time_saved <- as.integer(round(as_num(dat$num_observations)))
  stop_time_saved[!is.finite(stop_time_saved)] <- NA_integer_
  correct_action <- as.integer(round(as_num(dat$correct_action)))
  max_observations <- as.integer(round(as_num(dat$max_observations)))
  max_after <- max(step_ids, na.rm = TRUE)
  pieces <- vector("list", length(step_ids))
  piece_i <- 0L
  for (step in step_ids) {
    a_col <- sprintf("full_policy_choose_a_t%d", step)
    b_col <- sprintf("full_policy_choose_b_t%d", step)
    if (!all(c(a_col, b_col) %in% names(dat))) {
      next
    }
    relative_timestep <- step - stop_time_saved
    valid <- is.finite(stop_time_saved) &
      is.finite(max_observations) &
      step <= max_observations &
      correct_action %in% c(1L, 2L)
    keep <- valid &
      relative_timestep >= -max_steps_before_stop &
      relative_timestep <= max_after
    if (!any(keep)) {
      next
    }
    p_a <- as_num(dat[[a_col]])
    p_b <- as_num(dat[[b_col]])
    p_choice <- p_a + p_b
    p_target <- ifelse(correct_action == 1L, p_a, p_b)
    p_target_given_choice <- p_target / pmax(p_choice, 1e-12)
    piece_i <- piece_i + 1L
    pieces[[piece_i]] <- data.frame(
      family = dat$family[keep],
      parameter_value = as_num(dat$parameter_value[keep]),
      observation_noise_std = as_num(dat$observation_noise_std[keep]),
      run_id = dat$run_id[keep],
      trial_id = trial_id[keep],
      coherence = as_num(dat$coherence[keep]),
      timestep = step,
      stopping_timestep = stop_time_saved[keep],
      relative_timestep = relative_timestep[keep],
      p_target_given_choice = p_target_given_choice[keep],
      stringsAsFactors = FALSE
    )
  }
  if (piece_i == 0L) {
    return(data.frame())
  }
  rbind_fill(pieces[seq_len(piece_i)])
}

build_response_metric_data <- function(dat) {
  if (nrow(dat) == 0L) {
    return(data.frame())
  }
  metrics <- c(
    choice_aligned_cumulative_evidence = "Choice-aligned cumulative evidence",
    p_stop = "Probability of stopping",
    p_eventual_choice_given_stop = "Probability of eventual choice | stop"
  )
  pieces <- vector("list", length(metrics))
  piece_i <- 0L
  for (metric_col in names(metrics)) {
    if (!metric_col %in% names(dat)) {
      next
    }
    piece_i <- piece_i + 1L
    piece <- data.frame(
      family = dat$family,
      parameter_value = dat$parameter_value,
      observation_noise_std = dat$observation_noise_std,
      run_id = dat$run_id,
      trial_id = dat$trial_id,
      timestep = dat$timestep,
      relative_timestep = dat$relative_timestep,
      metric = metric_col,
      metric_label = metrics[[metric_col]],
      value = dat[[metric_col]],
      stringsAsFactors = FALSE
    )
    if ("coherence" %in% names(dat)) {
      piece$coherence <- as_num(dat$coherence)
    }
    pieces[[piece_i]] <- piece
  }
  if (piece_i == 0L) {
    return(data.frame())
  }
  rbind_fill(pieces[seq_len(piece_i)])
}

select_response_locked_coherence <- function(values, requested = NULL) {
  values <- as_num(values)
  values <- values[is.finite(values)]
  if (length(values) == 0L) {
    return(NA_real_)
  }
  if (!is.null(requested) && nzchar(trim_string(requested))) {
    requested_values <- as_num(parse_csv_values(requested))
    requested_values <- requested_values[is.finite(requested_values)]
    if (length(requested_values) == 0L) {
      stop("--response-locked-coherence must contain a numeric value.")
    }
    snapped <- snap_requested_values(requested_values[[1L]], values, "response-locked coherence", tol = 1e-5)
    if (length(snapped) == 0L) {
      stop(sprintf(
        "Requested response-locked coherence=%s was not found. Available: %s",
        num_label(requested_values[[1L]]),
        values_label(values)
      ))
    }
    return(snapped[[1L]])
  }
  nonzero <- values[abs(values) > 1e-12]
  if (length(nonzero) > 0L) {
    return(most_common_numeric(nonzero))
  }
  most_common_numeric(values)
}

make_quantile_bins <- function(values, n_bins = 20L) {
  values <- as_num(values)
  values <- values[is.finite(values)]
  if (length(values) == 0L) {
    stop("No cumulative-evidence data available for binning.")
  }
  probs <- seq(0, 1, length.out = n_bins + 1L)
  breaks <- unique(as.numeric(stats::quantile(values, probs = probs, na.rm = TRUE, names = FALSE, type = 8)))
  if (length(breaks) < 3L) {
    breaks <- unique(pretty(values, n = min(n_bins, 10L)))
  }
  if (length(breaks) < 3L) {
    stop("Could not construct at least two cumulative-evidence bins.")
  }
  breaks[1L] <- -Inf
  breaks[length(breaks)] <- Inf
  breaks
}

family_color_values <- function(family, params) {
  params <- sort(unique(as_num(params)))
  params <- params[is.finite(params)]
  if (length(params) == 0L) return(character())
  palette <- if (identical(family, "beta")) {
    grDevices::colorRampPalette(c("#74c476", "#238b45", "#00441b"))
  } else {
    grDevices::colorRampPalette(c("#6baed6", "#2171b5", "#08306b"))
  }
  cols <- palette(max(length(params), 2L))[seq_along(params)]
  names(cols) <- as.character(params)
  cols
}

series_pch <- function(family) if (identical(family, "beta")) 16 else 17

coherence_color_values <- function(values) {
  values <- sort(unique(as_num(values)))
  values <- values[is.finite(values)]
  if (length(values) == 0L) return(character())
  palette <- grDevices::colorRampPalette(c("#d9d9d9", "#969696", "#525252", "#111111"))
  cols <- palette(max(length(values), 2L))[seq_along(values)]
  names(cols) <- as.character(values)
  cols
}

target_panel_side_mm <- 33
target_panel_side_in <- target_panel_side_mm / 25.4
panel_margin_in <- c(bottom = 0.66, left = 0.78, top = 0.08, right = 0.12)
label_margin_in <- c(bottom = 0.02, left = 0.02, top = 0.02, right = 0.02)

draw_error_bars <- function(x, y, sem, col) {
  sem <- as_num(sem)
  keep <- is.finite(x) & is.finite(y) & is.finite(sem) & sem > 0
  if (any(keep)) {
    graphics::arrows(
      x[keep], y[keep] - sem[keep], x[keep], y[keep] + sem[keep],
      angle = 90, code = 3, length = 0.025, col = col, lwd = 0.7
    )
  }
}

safe_range <- function(values, pad_fraction = 0.05, fallback = c(0, 1)) {
  values <- as_num(values)
  values <- values[is.finite(values)]
  if (length(values) == 0L) return(fallback)
  lim <- range(values)
  if (abs(diff(lim)) < 1e-12) {
    lim <- lim + c(-0.5, 0.5)
  }
  pad <- diff(lim) * pad_fraction
  lim + c(-pad, pad)
}

relative_timestep_range <- function(values) {
  values <- as_num(values)
  values <- values[is.finite(values)]
  if (length(values) == 0L) {
    return(c(-1, 0))
  }
  c(min(values), max(0, max(values)))
}

shared_axis_limit <- function(summary_data, value_col, fixed_lim = NULL, include_sem = FALSE, fallback = c(0, 1)) {
  if (!is.null(fixed_lim)) {
    return(fixed_lim)
  }
  values <- summary_data[[value_col]]
  if (include_sem) {
    sem_col <- paste0(value_col, "_sem")
    if (sem_col %in% names(summary_data)) {
      values <- c(values, summary_data[[value_col]] - summary_data[[sem_col]], summary_data[[value_col]] + summary_data[[sem_col]])
    }
  }
  safe_range(values, fallback = fallback)
}

nonnegative_axis_limit <- function(summary_data, value_col, fallback = c(0, 1)) {
  lim <- shared_axis_limit(summary_data, value_col, include_sem = TRUE, fallback = fallback)
  upper <- max(lim[[2L]], fallback[[2L]], na.rm = TRUE)
  if (!is.finite(upper) || upper <= 0) upper <- fallback[[2L]]
  c(0, upper)
}

positive_log_axis_limit <- function(summary_data, value_col, fallback = c(1e-6, 1)) {
  values <- as_num(summary_data[[value_col]])
  sem_col <- paste0(value_col, "_sem")
  if (sem_col %in% names(summary_data)) {
    sem <- as_num(summary_data[[sem_col]])
    values <- c(values, as_num(summary_data[[value_col]]) - sem, as_num(summary_data[[value_col]]) + sem)
  }
  values <- values[is.finite(values) & values > 0]
  if (length(values) == 0L) return(fallback)
  lim <- c(min(values) / 10, max(values) * 1.15)
  if (abs(diff(log10(lim))) < 1e-8) {
    lim <- lim * c(0.5, 2)
  }
  lim[!is.finite(lim) | lim <= 0] <- fallback[!is.finite(lim) | lim <= 0]
  lim
}

signed_log_axis_limit <- function(summary_data, value_col, fallback = c(-1, 1)) {
  values <- as_num(summary_data[[value_col]])
  sem_col <- paste0(value_col, "_sem")
  if (sem_col %in% names(summary_data)) {
    sem <- as_num(summary_data[[sem_col]])
    values <- c(values, as_num(summary_data[[value_col]]) - sem, as_num(summary_data[[value_col]]) + sem)
  }
  values <- values[is.finite(values)]
  if (length(values) == 0L) return(fallback)
  max_abs <- max(abs(values), na.rm = TRUE)
  if (!is.finite(max_abs) || max_abs <= 0) return(fallback)
  c(-1, 1) * max_abs * 1.05
}

y_scale_floor <- function(summary_data, value_col, y_scale = "linear") {
  values <- as_num(summary_data[[value_col]])
  sem_col <- paste0(value_col, "_sem")
  if (sem_col %in% names(summary_data)) {
    sem <- as_num(summary_data[[sem_col]])
    values <- c(values, as_num(summary_data[[value_col]]) - sem, as_num(summary_data[[value_col]]) + sem)
  }
  if (identical(y_scale, "signed_log10")) {
    values <- abs(values)
  }
  values <- values[is.finite(values) & values > 0]
  if (length(values) == 0L) {
    return(1e-8)
  }
  max(min(values, na.rm = TRUE) / 10, 1e-12)
}

transform_y_values <- function(values, y_scale = "linear", floor_value = 1e-8) {
  values <- as_num(values)
  if (identical(y_scale, "log10")) {
    out <- rep(NA_real_, length(values))
    keep <- is.finite(values)
    out[keep] <- log10(pmax(values[keep], floor_value))
    return(out)
  }
  if (identical(y_scale, "signed_log10")) {
    return(sign(values) * log10(1 + abs(values) / max(floor_value, 1e-12)))
  }
  values
}

log_axis_ticks <- function(raw_lim) {
  raw_lim <- as_num(raw_lim)
  raw_lim <- raw_lim[is.finite(raw_lim) & raw_lim > 0]
  if (length(raw_lim) == 0L) return(numeric())
  exp_seq <- seq.int(floor(log10(min(raw_lim))), ceiling(log10(max(raw_lim))), by = 1L)
  ticks <- 10^exp_seq
  ticks[ticks >= min(raw_lim) & ticks <= max(raw_lim)]
}

signed_log_axis_ticks <- function(raw_lim, floor_value = 1e-8) {
  raw_lim <- as_num(raw_lim)
  raw_lim <- raw_lim[is.finite(raw_lim)]
  if (length(raw_lim) == 0L) return(0)
  max_abs <- max(abs(raw_lim), na.rm = TRUE)
  if (!is.finite(max_abs) || max_abs <= 0) return(0)
  min_abs <- max(floor_value, 1e-12)
  exp_seq <- seq.int(floor(log10(min_abs)), ceiling(log10(max_abs)), by = 1L)
  pos_ticks <- 10^exp_seq
  pos_ticks <- pos_ticks[pos_ticks <= max_abs & pos_ticks >= min_abs]
  unique(c(-rev(pos_ticks), 0, pos_ticks))
}

draw_scaled_y_axis <- function(raw_lim, y_scale = "linear", floor_value = 1e-8) {
  if (identical(y_scale, "log10")) {
    ticks <- log_axis_ticks(raw_lim)
    if (length(ticks) > 0L) {
      axis(2, at = transform_y_values(ticks, y_scale, floor_value), labels = vapply(ticks, num_label, character(1)))
    }
    return(invisible(NULL))
  }
  if (identical(y_scale, "signed_log10")) {
    ticks <- signed_log_axis_ticks(raw_lim, floor_value)
    axis(2, at = transform_y_values(ticks, y_scale, floor_value), labels = vapply(ticks, num_label, character(1)))
    return(invisible(NULL))
  }
  axis(2)
  invisible(NULL)
}

draw_scaled_error_bars <- function(x, y, sem, col, y_scale = "linear", floor_value = 1e-8) {
  sem <- as_num(sem)
  keep <- is.finite(x) & is.finite(y) & is.finite(sem) & sem > 0
  if (!any(keep)) return(invisible(NULL))
  lo <- y[keep] - sem[keep]
  hi <- y[keep] + sem[keep]
  if (identical(y_scale, "log10")) {
    hi <- pmax(hi, floor_value)
    lo <- pmax(lo, floor_value)
  }
  graphics::arrows(
    x[keep],
    transform_y_values(lo, y_scale, floor_value),
    x[keep],
    transform_y_values(hi, y_scale, floor_value),
    angle = 90, code = 3, length = 0.025, col = col, lwd = 0.7
  )
  invisible(NULL)
}

plot_curve_panel <- function(summary_data, family, x_col, y_col, xlab, ylab, main = "",
                             colors, params, y_lim = NULL, x_lim = NULL,
                             hline = NULL, vline = NULL,
                             y_scale = "linear", y_floor = NULL) {
  old_xpd <- par("xpd")
  on.exit(par(xpd = old_xpd), add = TRUE)
  par(xpd = FALSE)
  panel_data <- summary_data[summary_data$family == family, , drop = FALSE]
  if (is.null(x_lim)) x_lim <- safe_range(panel_data[[x_col]])
  if (is.null(y_floor)) {
    y_floor <- y_scale_floor(summary_data, y_col, y_scale)
  }
  raw_y_lim <- y_lim
  if (is.null(raw_y_lim) && identical(y_scale, "log10")) {
    raw_y_lim <- positive_log_axis_limit(summary_data, y_col)
  } else if (is.null(raw_y_lim) && identical(y_scale, "signed_log10")) {
    raw_y_lim <- signed_log_axis_limit(summary_data, y_col)
  } else if (is.null(raw_y_lim)) {
    sem_col <- paste0(y_col, "_sem")
    raw_y_lim <- safe_range(c(panel_data[[y_col]], panel_data[[y_col]] - panel_data[[sem_col]], panel_data[[y_col]] + panel_data[[sem_col]]))
  } else if (identical(y_scale, "signed_log10")) {
    max_abs <- max(abs(raw_y_lim), na.rm = TRUE)
    raw_y_lim <- c(-max_abs, max_abs)
  }
  plot_y_lim <- transform_y_values(raw_y_lim, y_scale, y_floor)
  if (!all(is.finite(plot_y_lim))) {
    plot_y_lim <- c(0, 1)
  }
  is_timestep_axis <- x_col %in% c("timestep_before_stopping", "relative_timestep", "stopping_timestep")
  plot_args <- list(
    x = NA,
    xlim = x_lim,
    ylim = plot_y_lim,
    xlab = xlab,
    ylab = ylab,
    cex.lab = 1,
    cex.axis = 1,
    cex.main = 1
  )
  if (!identical(y_scale, "linear")) {
    plot_args$yaxt <- "n"
  }
  if (is_timestep_axis) {
    plot_args$xaxt <- "n"
  }
  if (nzchar(trim_string(main))) {
    plot_args$main <- main
  }
  do.call(plot, plot_args)
  if (!identical(y_scale, "linear")) {
    draw_scaled_y_axis(raw_y_lim, y_scale, y_floor)
  }
  if (is_timestep_axis) {
    axis_ticks <- seq.int(ceiling(x_lim[[1L]]), floor(x_lim[[2L]]), by = 1L)
    if (length(axis_ticks) > 0L) {
      axis(1, at = axis_ticks)
    }
  }
  grid(col = "grey90")
  if (!is.null(hline)) {
    hline_plot <- transform_y_values(hline, y_scale, y_floor)
    hline_plot <- hline_plot[is.finite(hline_plot)]
    if (length(hline_plot) > 0L) {
      abline(h = hline_plot, col = "grey55", lty = 2, lwd = 0.8)
    }
  }
  if (!is.null(vline)) abline(v = vline, col = "grey55", lty = 2, lwd = 0.8)
  for (param in params) {
    line_data <- panel_data[parameter_equal(panel_data$parameter_value, param), , drop = FALSE]
    if (nrow(line_data) == 0L) next
    line_data <- line_data[order(as_num(line_data[[x_col]])), , drop = FALSE]
    x <- as_num(line_data[[x_col]])
    y <- as_num(line_data[[y_col]])
    y_plot <- transform_y_values(y, y_scale, y_floor)
    sem_col <- paste0(y_col, "_sem")
    col <- colors[[as.character(param)]]
    lines(x, y_plot, col = col, lwd = 1.3)
    points(x, y_plot, col = col, pch = series_pch(family), cex = 0.75)
    if (sem_col %in% names(line_data)) {
      draw_scaled_error_bars(x, y, line_data[[sem_col]], col, y_scale, y_floor)
    }
  }
  box()
}

draw_family_legend <- function(family, params, colors, title) {
  plot.new()
  params <- sort(unique(as_num(params)))
  params <- params[is.finite(params)]
  if (length(params) == 0L) return(invisible(NULL))
  labels <- if (identical(family, "beta")) {
    paste0("beta ", vapply(params, num_label, character(1)))
  } else {
    paste0("opp ", vapply(params, num_label, character(1)))
  }
  graphics::legend(
    "center",
    legend = labels,
    title = title,
    col = colors[as.character(params)],
    pch = series_pch(family),
    lwd = 1.3,
    bty = "n",
    cex = 1,
    pt.cex = 0.9
  )
}

draw_family_header <- function(family, params, colors, header, legend_title) {
  plot.new()
  text(0.5, 0.82, header, cex = 1)
  params <- sort(unique(as_num(params)))
  params <- params[is.finite(params)]
  if (length(params) == 0L) return(invisible(NULL))
  labels <- if (identical(family, "beta")) {
    paste0("beta ", vapply(params, num_label, character(1)))
  } else {
    paste0("opp ", vapply(params, num_label, character(1)))
  }
  graphics::legend(
    "bottom",
    legend = labels,
    title = legend_title,
    col = colors[as.character(params)],
    pch = series_pch(family),
    lwd = 1.3,
    bty = "n",
    cex = 0.9,
    pt.cex = 0.85,
    horiz = TRUE,
    x.intersp = 0.8,
    y.intersp = 0.9
  )
  invisible(NULL)
}

draw_obsstd_label <- function(obsstd) {
  plot.new()
  text(
    0.5, 0.5,
    sprintf("std = %s", num_label(obsstd)),
    srt = 90,
    cex = 1
  )
  invisible(NULL)
}

draw_side_label <- function(label) {
  plot.new()
  text(0.5, 0.5, label, srt = 90, cex = 1)
  invisible(NULL)
}

if (!dir.exists(input_dir)) {
  stop(sprintf("Simulation input directory not found: %s", input_dir))
}

file_index <- parse_evidence_filename_index(input_dir)
if (nrow(file_index) == 0L) {
  stop(sprintf("No evidence simulation CSVs were found under %s.", input_dir))
}
file_index <- file_index[file_index$input_type == input_type, , drop = FALSE]
file_index <- filter_numeric_option(file_index, "loss_scale", option_loss_scale$value, "--loss-scale")
file_index <- filter_numeric_option(file_index, "alpha", option_alpha$value, "--alpha")
file_index <- filter_numeric_option(file_index, "rnn_units", option_rnn$value, "--rnn-units")
file_index <- filter_numeric_option(file_index, "latent_dim", option_latent$value, "--latent-dim")
file_index <- filter_numeric_option(file_index, "max_observations", option_maxobs$value, "--max-observations")
file_index <- filter_numeric_option(file_index, "correct_reward", option_correct_reward$value, "--correct-reward")
file_index <- file_index[file_index$pay_kl_on_stop == pay_kl_on_stop_mode, , drop = FALSE]
file_index <- file_index[file_index$choice_at_end_only == observer_only_mode, , drop = FALSE]
if (nrow(file_index) == 0L) {
  stop(sprintf(
    "No evidence simulation CSVs remain after metadata, correct-reward, stop-paid, and observer-only filters. Requested correct_reward=%s; pay_kl_on_stop=%s; observer_only=%s.",
    if (is.null(option_correct_reward$value)) "<none>" else option_correct_reward$value,
    if (pay_kl_on_stop_mode) "true" else "false",
    if (observer_only_mode) "true" else "false"
  ))
}

if (!is.null(option_obsstd$value) && nzchar(trim_string(option_obsstd$value))) {
  requested_obsstd <- as_num(parse_csv_values(option_obsstd$value))
  requested_obsstd <- requested_obsstd[is.finite(requested_obsstd)]
  if (length(requested_obsstd) == 0L) {
    stop("--observation-noise-std must contain numeric value(s).")
  }
  file_index <- filter_numeric_option(file_index, "observation_noise_std", option_obsstd$value, "--observation-noise-std")
  if (nrow(file_index) == 0L) {
    stop("No evidence simulation CSVs remain after observation-noise filter.")
  }
  selected_obsstd_values <- sort(unique(file_index$observation_noise_std[is.finite(file_index$observation_noise_std)]))
  missing_obsstd <- requested_obsstd[!vapply(requested_obsstd, function(v) any(parameter_equal(file_index$observation_noise_std, v)), logical(1))]
  if (length(missing_obsstd) > 0L) {
    warning(sprintf(
      "Requested observation noise std value(s) not found after filtering: %s",
      values_label(missing_obsstd)
    ))
  }
} else {
  selected_obsstd <- most_common_numeric(file_index$observation_noise_std)
  file_index <- file_index[parameter_equal(file_index$observation_noise_std, selected_obsstd), , drop = FALSE]
  selected_obsstd_values <- selected_obsstd
  message(sprintf("No observation noise was supplied; selected most common obsstd=%s.", num_label(selected_obsstd)))
}

fixed_opp <- select_fixed_value(file_index$opportunity, option_fixed_opp$value, preferred = 0.0, label = "opportunity")
fixed_memory_lambda <- select_fixed_value(
  file_index$memory_lambda,
  if (!is.null(option_fixed_memory_lambda$value)) option_fixed_memory_lambda$value else option_fixed_beta$value,
  preferred = 0.0,
  label = "memory_lambda"
)

requested_beta_values <- if (!is.null(option_memory_lambda_values$value)) {
  as_num(parse_csv_values(option_memory_lambda_values$value))
} else if (!is.null(option_beta_values$value)) {
  as_num(parse_csv_values(option_beta_values$value))
} else {
  numeric()
}
requested_opp_values <- if (!is.null(option_opp_values$value)) as_num(parse_csv_values(option_opp_values$value)) else numeric()
available_beta_values <- sort(unique(file_index$memory_lambda[parameter_equal(file_index$opportunity, fixed_opp)]))
available_opp_values <- sort(unique(file_index$opportunity[parameter_equal(file_index$memory_lambda, fixed_memory_lambda)]))
beta_values <- if (length(requested_beta_values) > 0L) requested_beta_values else available_beta_values
opportunity_values <- if (length(requested_opp_values) > 0L) requested_opp_values else available_opp_values
beta_values <- sort(unique(beta_values[is.finite(beta_values)]))
opportunity_values <- sort(unique(opportunity_values[is.finite(opportunity_values)]))

keep_beta <- rep(FALSE, nrow(file_index))
memory_seed_keep <- numeric_option_mask(file_index, "seed", memory_seed_filter, "memory-lambda seed_arg/--seeds")
opportunity_seed_keep <- numeric_option_mask(file_index, "seed", opportunity_seed_filter, "opportunity seed_arg/--seeds")
memory_expansion_keep <- string_option_mask(file_index, "expansion", memory_expansion_filter, "memory-lambda expansion_decision_version")
opportunity_expansion_keep <- string_option_mask(file_index, "expansion", opportunity_expansion_filter, "opportunity expansion_decision_version")
memory_variant_keep <- string_option_mask(file_index, "variant", memory_variant_filter, "memory-lambda model_variant")
opportunity_variant_keep <- string_option_mask(file_index, "variant", opportunity_variant_filter, "opportunity model_variant")
for (b in beta_values) {
  keep_beta <- keep_beta |
    (
      parameter_equal(file_index$memory_lambda, b) &
        parameter_equal(file_index$opportunity, fixed_opp) &
        memory_seed_keep &
        memory_expansion_keep &
        memory_variant_keep
    )
}
keep_opp <- rep(FALSE, nrow(file_index))
for (o in opportunity_values) {
  keep_opp <- keep_opp |
    (
      parameter_equal(file_index$opportunity, o) &
        parameter_equal(file_index$memory_lambda, fixed_memory_lambda) &
        opportunity_seed_keep &
        opportunity_expansion_keep &
        opportunity_variant_keep
    )
}
manifest <- file_index[keep_beta | keep_opp, , drop = FALSE]
if (nrow(manifest) == 0L) {
  stop("No files match the requested memory-lambda/opportunity comparisons.")
}

message(sprintf("Loaded manifest with %d evidence simulation file(s).", nrow(manifest)))
message(sprintf("Available beta values after filters: %s", values_label(file_index$beta)))
message(sprintf("Available memory-lambda values after filters: %s", values_label(file_index$memory_lambda)))
message(sprintf("Available opportunity-cost values after filters: %s", values_label(file_index$opportunity)))
message(sprintf("Available observation noise std values after filters: %s", values_label(file_index$observation_noise_std)))
message(sprintf("Available correct-reward values after filters: %s", values_label(file_index$correct_reward)))
message(sprintf("Observation noise std values included in the figure: %s", values_label(selected_obsstd_values)))
message(sprintf("Correct terminal reward scale: %s", values_label(file_index$correct_reward)))
message(sprintf("Stop-paid KL file mode: %s", if (pay_kl_on_stop_mode) "using _stop_paid CSVs" else "using legacy/no _stop_paid CSVs"))
message(sprintf("Observer-only file mode: %s", if (observer_only_mode) "using _observer_endchoice CSVs" else "using non-observer CSVs"))
message(sprintf("Minimum samples per point: all plots=%d; delta plots=%d", minimum_samples, delta_minimum_samples))
message(sprintf("Selected fixed opportunity cost for memory-lambda comparison: %s", num_label(fixed_opp)))
message(sprintf("Selected fixed memory lambda for opportunity comparison: %s", num_label(fixed_memory_lambda)))
message(sprintf(
  "Seed filter: memory-lambda=%s; opportunity=%s",
  if (is.null(memory_seed_filter) || !nzchar(trim_string(memory_seed_filter))) "<all>" else memory_seed_filter,
  if (is.null(opportunity_seed_filter) || !nzchar(trim_string(opportunity_seed_filter))) "<all>" else opportunity_seed_filter
))

loaded <- vector("list", nrow(manifest))
for (i in seq_len(nrow(manifest))) {
  loaded[[i]] <- load_one_evidence_file(manifest$file[[i]], manifest[i, , drop = FALSE])
}
trial_data <- rbind_fill(loaded)
trial_data$family <- ifelse(parameter_equal(trial_data$opportunity, fixed_opp), "beta", "opportunity")
trial_data$parameter_value <- ifelse(trial_data$family == "beta", trial_data$memory_lambda, trial_data$opportunity)
trial_data$parameter_label <- ifelse(
  trial_data$family == "beta",
  paste0("lambda=", vapply(trial_data$parameter_value, num_label, character(1))),
  paste0("opp=", vapply(trial_data$parameter_value, num_label, character(1)))
)
trial_data <- trial_data[is.finite(trial_data$coherence) & is.finite(trial_data$signed_coherence), , drop = FALSE]
if (length(unique(trial_data$signed_coherence[is.finite(trial_data$signed_coherence)])) < 2L) {
  stop("Fewer than two signed-coherence conditions are available.")
}
if (!any(is.finite(trial_data$num_observations))) {
  stop("No valid stopping-time observations were found.")
}
if (!any(is.finite(trial_data$decision_cumulative_evidence))) {
  stop("No valid cumulative-evidence observations were found.")
}

message(sprintf("Total trials loaded: %d", nrow(trial_data)))
message(sprintf("Available coherence magnitudes: %s", values_label(trial_data$coherence)))
delta_coherence_values <- snap_requested_values(
  delta_coherence_values,
  trial_data$coherence,
  "coherence",
  tol = 1e-5
)
simple_coherence_values <- snap_requested_values(
  simple_coherence_values,
  trial_data$coherence,
  "simple coherence",
  tol = 1e-5
)
message(sprintf("Independent run/file count: %d", length(unique(trial_data$run_id))))
message(sprintf(
  "Trials retained in memory-lambda comparison: %d; opportunity comparison: %d",
  sum(trial_data$family == "beta"),
  sum(trial_data$family == "opportunity")
))

psychometric_summary <- summarize_across_runs(
  trial_data,
  c("family", "parameter_value", "observation_noise_std", "signed_coherence"),
  "choose_right"
)
chronometric_summary <- summarize_across_runs(
  trial_data,
  c("family", "parameter_value", "observation_noise_std", "coherence"),
  "num_observations"
)
total_kl_summary <- summarize_across_runs(
  trial_data,
  c("family", "parameter_value", "observation_noise_std", "coherence"),
  "total_kl_paid"
)
accuracy_summary <- summarize_across_runs(
  trial_data,
  c("family", "parameter_value", "observation_noise_std", "coherence"),
  "choose_correct"
)
accuracy_tradeoff_summary <- summarize_across_runs(
  trial_data,
  c("family", "parameter_value", "observation_noise_std", "coherence"),
  "choose_correct",
  x_cols = c("total_kl_paid", "num_observations")
)
kl_time_tradeoff_summary <- summarize_across_runs(
  trial_data,
  c("family", "parameter_value", "observation_noise_std", "coherence"),
  "total_kl_paid",
  x_cols = "num_observations"
)

breaks <- make_quantile_bins(trial_data$decision_cumulative_evidence, n_bins = n_bins)
trial_data$evidence_bin <- cut(trial_data$decision_cumulative_evidence, breaks = breaks, include.lowest = TRUE, right = TRUE)
realized_data <- trial_data[!is.na(trial_data$evidence_bin), , drop = FALSE]
realized_summary <- summarize_across_runs(
  realized_data,
  c("family", "parameter_value", "observation_noise_std", "evidence_bin"),
  "choose_right",
  x_cols = "decision_cumulative_evidence"
)
timestep_data <- build_timestep_data(trial_data)
if (nrow(timestep_data) == 0L) {
  warning("No per-timestep policy/KL columns found; skipping terminal-choice-policy and KL-by-timestep plots.")
}
terminal_choice_logit_summary <- summarize_across_runs(
  timestep_data,
  c("family", "parameter_value", "observation_noise_std", "timestep_before_stopping"),
  "terminal_choice_logit_correct"
)
kl_timestep_summary <- summarize_across_runs(
  timestep_data,
  c("family", "parameter_value", "observation_noise_std", "timestep_before_stopping"),
  "kl_paid"
)
kl_timestep_by_coherence_summary <- summarize_across_runs(
  timestep_data,
  c("family", "parameter_value", "observation_noise_std", "coherence", "timestep_before_stopping"),
  "kl_paid",
  min_samples = delta_minimum_samples
)
delta_action_logit_summary <- summarize_across_runs(
  timestep_data,
  c("family", "parameter_value", "observation_noise_std", "timestep_before_stopping"),
  "abs_delta_action_aligned_action_logit",
  min_samples = delta_minimum_samples
)
delta_z_mu_summary <- summarize_across_runs(
  timestep_data,
  c("family", "parameter_value", "observation_noise_std", "timestep_before_stopping"),
  "abs_delta_z_mu",
  min_samples = delta_minimum_samples
)
delta_z_sigma_summary <- summarize_across_runs(
  timestep_data,
  c("family", "parameter_value", "observation_noise_std", "timestep_before_stopping"),
  "signed_delta_z_sigma",
  min_samples = delta_minimum_samples
)
z_sigma_timestep_summary <- summarize_across_runs(
  timestep_data,
  c("family", "parameter_value", "observation_noise_std", "timestep_before_stopping"),
  "average_z_sigma",
  min_samples = delta_minimum_samples
)
delta_action_logit_by_coherence_summary <- summarize_across_runs(
  timestep_data,
  c("family", "parameter_value", "observation_noise_std", "coherence", "timestep_before_stopping"),
  "abs_delta_action_aligned_action_logit",
  min_samples = delta_minimum_samples
)
delta_z_mu_by_coherence_summary <- summarize_across_runs(
  timestep_data,
  c("family", "parameter_value", "observation_noise_std", "coherence", "timestep_before_stopping"),
  "abs_delta_z_mu",
  min_samples = delta_minimum_samples
)
delta_z_sigma_by_coherence_summary <- summarize_across_runs(
  timestep_data,
  c("family", "parameter_value", "observation_noise_std", "coherence", "timestep_before_stopping"),
  "signed_delta_z_sigma",
  min_samples = delta_minimum_samples
)
z_sigma_timestep_by_coherence_summary <- summarize_across_runs(
  timestep_data,
  c("family", "parameter_value", "observation_noise_std", "coherence", "timestep_before_stopping"),
  "average_z_sigma",
  min_samples = delta_minimum_samples
)
message(sprintf(
  "Cumulative-evidence curve uses %d pooled quantile bin(s) after duplicate-break removal.",
  length(unique(realized_data$evidence_bin))
))
if (length(delta_coherence_values) > 0L) {
  message(sprintf(
    "Writing coherence-specific delta and response-locked cumulative-evidence plots for coherence: %s",
    values_label(delta_coherence_values)
  ))
}

response_locked_coherence <- select_response_locked_coherence(trial_data$coherence, option_response_locked_coherence$value)
response_trial_data <- trial_data[parameter_equal(trial_data$coherence, response_locked_coherence, tol = 1e-5), , drop = FALSE]
response_locked_data <- build_response_locked_data(response_trial_data, max_steps_before_stop = max_steps_before_stop)
response_locked_by_coherence_data <- build_response_locked_data(trial_data, max_steps_before_stop = max_steps_before_stop)
full_policy_target_response_data <- build_full_policy_target_response_data(
  trial_data,
  max_steps_before_stop = max_steps_before_stop
)
if (nrow(response_locked_data) > 0L) {
  response_trial_level <- response_locked_data[!duplicated(response_locked_data$trial_id), , drop = FALSE]
  forced_n <- sum(response_trial_level$forced_terminal, na.rm = TRUE)
  message(sprintf(
    "Response-locked plots use coherence=%s and relative timesteps -%d,...,0.",
    num_label(response_locked_coherence),
    max_steps_before_stop
  ))
  message(sprintf(
    "Response-locked forced-terminal trials: %d/%d (%.2f%%).",
    forced_n,
    nrow(response_trial_level),
    100 * forced_n / max(nrow(response_trial_level), 1L)
  ))
  if (!include_forced_stops) {
    response_locked_data <- response_locked_data[!response_locked_data$forced_terminal, , drop = FALSE]
  }
  message(sprintf(
    "Response-locked forced terminal decisions %s.",
    if (include_forced_stops) "included" else "excluded"
  ))
} else {
  warning("No response-locked trajectory rows could be reconstructed; skipping response-locked plots.")
}
if (!include_forced_stops && nrow(response_locked_by_coherence_data) > 0L) {
  response_locked_by_coherence_data <- response_locked_by_coherence_data[
    !response_locked_by_coherence_data$forced_terminal,
    ,
    drop = FALSE
  ]
}
response_metric_data <- build_response_metric_data(response_locked_data)
response_locked_summary <- summarize_across_runs(
  response_metric_data,
  c("family", "parameter_value", "observation_noise_std", "relative_timestep", "metric", "metric_label"),
  "value"
)
response_locked_by_coherence_metric_data <- build_response_metric_data(response_locked_by_coherence_data)
response_locked_by_coherence_summary <- summarize_across_runs(
  response_locked_by_coherence_metric_data,
  c("family", "parameter_value", "observation_noise_std", "coherence", "relative_timestep", "metric", "metric_label"),
  "value"
)
chronological_by_coherence_summary <- summarize_across_runs(
  response_locked_by_coherence_metric_data,
  c("family", "parameter_value", "observation_noise_std", "coherence", "timestep", "metric", "metric_label"),
  "value"
)
full_policy_target_response_summary <- summarize_across_runs(
  full_policy_target_response_data,
  c("family", "parameter_value", "observation_noise_std", "coherence", "relative_timestep"),
  "p_target_given_choice",
  min_samples = delta_minimum_samples
)
threshold_stop_data <- response_locked_data[
  response_locked_data$relative_timestep == 0 &
    is.finite(response_locked_data$choice_aligned_cumulative_evidence) &
    is.finite(response_locked_data$stopping_timestep),
  ,
  drop = FALSE
]
threshold_by_duration_summary <- if (!observer_only_mode && nrow(threshold_stop_data) > 0L) {
  summarize_across_runs(
    threshold_stop_data,
    c("family", "parameter_value", "observation_noise_std", "stopping_timestep"),
    "choice_aligned_cumulative_evidence",
    min_samples = delta_minimum_samples
  )
} else {
  data.frame()
}
threshold_by_coherence_stop_data <- response_locked_by_coherence_data[
  response_locked_by_coherence_data$relative_timestep == 0 &
    is.finite(response_locked_by_coherence_data$choice_aligned_cumulative_evidence) &
    is.finite(response_locked_by_coherence_data$stopping_timestep),
  ,
  drop = FALSE
]
threshold_by_duration_coherence_summary <- if (!observer_only_mode && nrow(threshold_by_coherence_stop_data) > 0L) {
  summarize_across_runs(
    threshold_by_coherence_stop_data,
    c("family", "parameter_value", "observation_noise_std", "coherence", "stopping_timestep"),
    "choice_aligned_cumulative_evidence",
    min_samples = delta_minimum_samples
  )
} else {
  data.frame()
}
if (observer_only_mode) {
  message("Skipping decision-threshold-by-duration plots for observer-only runs because stopping duration is fixed by design.")
}
if (!observer_only_mode) {
  message("Decision-threshold-by-duration plots are disabled; using response-locked target-choice probability instead.")
}
threshold_by_duration_summary <- data.frame()
threshold_by_duration_coherence_summary <- data.frame()

beta_colors <- family_color_values("beta", beta_values)
opp_colors <- family_color_values("opportunity", opportunity_values)

if (!is.null(option_output_file$value) && nzchar(trim_string(option_output_file$value))) {
  output_file_stem <- sub("\\.png$", "", option_output_file$value, ignore.case = TRUE)
  output_dir <- dirname(output_file_stem)
  if (!nzchar(output_dir) || identical(output_dir, ".")) {
    output_dir <- "."
  }
} else {
  observer_output_dir <- if (observer_only_mode) "observer_only" else "no_observer"
  output_dir <- file.path(output_root, "evidence_accumulation_compare", observer_output_dir)
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  output_file_stem <- file.path(
    output_dir,
    sprintf(
      "evidence_accumulation_beta_opp_comparison_obsstd_%s%s",
      values_token(selected_obsstd_values),
      paste0(
        "_correctreward_",
        values_token(unique(file_index$correct_reward[is.finite(file_index$correct_reward)])),
        if (pay_kl_on_stop_mode) "_stop_paid" else ""
      )
    )
  )
}
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

prob_ylim <- c(0, 1)
signed_xlim <- safe_range(trial_data$signed_coherence)
coh_xlim <- safe_range(trial_data$coherence, fallback = c(0, 1))
evidence_xlim <- safe_range(realized_summary$decision_cumulative_evidence)

analysis_specs <- list(
  list(
    name = "Psychometric",
    slug = "psychometric",
    summary = psychometric_summary,
    x_col = "signed_coherence",
    y_col = "choose_right",
    xlab = "Signed coherence",
    ylab = "Probability of choosing right",
    y_lim = prob_ylim,
    x_lim = signed_xlim,
    hline = 0.5,
    vline = 0
  ),
  list(
    name = "Chronometric",
    slug = "chronometric",
    summary = chronometric_summary,
    x_col = "coherence",
    y_col = "num_observations",
    xlab = "Coherence magnitude",
    ylab = "Number of observations before choice",
    y_lim = NULL,
    x_lim = coh_xlim,
    hline = NULL,
    vline = NULL
  ),
  list(
    name = "Total paid KL",
    slug = "total_paid_kl",
    summary = total_kl_summary,
    x_col = "coherence",
    y_col = "total_kl_paid",
    xlab = "Coherence magnitude",
    ylab = "Total paid KL",
    y_lim = positive_log_axis_limit(total_kl_summary, "total_kl_paid"),
    x_lim = coh_xlim,
    hline = NULL,
    vline = NULL,
    y_scale = "log10"
  ),
  list(
    name = "Accuracy",
    slug = "accuracy",
    summary = accuracy_summary,
    x_col = "coherence",
    y_col = "choose_correct",
    xlab = "Coherence magnitude",
    ylab = "Choice accuracy",
    y_lim = prob_ylim,
    x_lim = coh_xlim,
    hline = 0.5,
    vline = NULL
  ),
  list(
    name = "Accuracy vs total paid KL",
    slug = "accuracy_vs_total_paid_kl",
    summary = accuracy_tradeoff_summary,
    x_col = "total_kl_paid",
    y_col = "choose_correct",
    xlab = "Average total\npaid KL",
    ylab = "Choice accuracy",
    y_lim = prob_ylim,
    x_lim = nonnegative_axis_limit(accuracy_tradeoff_summary, "total_kl_paid"),
    hline = 0.5,
    vline = NULL
  ),
  list(
    name = "Accuracy vs timesteps before stopping",
    slug = "accuracy_vs_timestep_before_stop",
    summary = accuracy_tradeoff_summary,
    x_col = "num_observations",
    y_col = "choose_correct",
    xlab = "Average timesteps\nbefore stopping",
    ylab = "Choice accuracy",
    y_lim = prob_ylim,
    x_lim = nonnegative_axis_limit(accuracy_tradeoff_summary, "num_observations"),
    hline = 0.5,
    vline = NULL
  ),
  list(
    name = "Total paid KL vs timesteps before stopping",
    slug = "total_paid_kl_vs_timestep_before_stop",
    summary = kl_time_tradeoff_summary,
    x_col = "num_observations",
    y_col = "total_kl_paid",
    xlab = "Average timesteps\nbefore stopping",
    ylab = "Total paid KL",
    y_lim = positive_log_axis_limit(kl_time_tradeoff_summary, "total_kl_paid"),
    x_lim = nonnegative_axis_limit(kl_time_tradeoff_summary, "num_observations"),
    hline = NULL,
    vline = NULL,
    y_scale = "log10"
  ),
  list(
    name = "Realized evidence",
    slug = "realized_evidence",
    summary = realized_summary,
    x_col = "decision_cumulative_evidence",
    y_col = "choose_right",
    xlab = "Cumulative evidence at decision",
    ylab = "Probability of choosing right",
    y_lim = prob_ylim,
    x_lim = evidence_xlim,
    hline = 0.5,
    vline = 0
  )
)

if (nrow(terminal_choice_logit_summary) > 0L) {
  analysis_specs <- c(
    analysis_specs,
    list(
      list(
        name = "Terminal choice logit by timestep",
        slug = "terminal_choice_policy_by_timestep",
        summary = terminal_choice_logit_summary,
        x_col = "timestep_before_stopping",
        y_col = "terminal_choice_logit_correct",
        xlab = "Steps relative\nto stopping",
        ylab = "Terminal choice logit\nlog(P correct / P incorrect)",
        y_lim = NULL,
        x_lim = relative_timestep_range(terminal_choice_logit_summary$timestep_before_stopping),
        hline = 0,
        vline = NULL
      )
    )
  )
}

if (nrow(kl_timestep_summary) > 0L) {
  analysis_specs <- c(
    analysis_specs,
    list(
      list(
        name = "KL paid by timestep",
        slug = "kl_paid_by_timestep",
        summary = kl_timestep_summary,
        x_col = "timestep_before_stopping",
        y_col = "kl_paid",
        xlab = "Steps relative\nto stopping",
        ylab = "KL paid",
        y_lim = positive_log_axis_limit(kl_timestep_summary, "kl_paid"),
        x_lim = relative_timestep_range(kl_timestep_summary$timestep_before_stopping),
        hline = NULL,
        vline = NULL,
        y_scale = "log10"
      )
    )
  )
}

if (nrow(delta_action_logit_summary) > 0L) {
  analysis_specs <- c(
    analysis_specs,
    list(
      list(
        name = "Abs delta action-aligned terminal logit by timestep",
        slug = "abs_delta_action_aligned_action_logit_by_timestep",
        summary = delta_action_logit_summary,
        x_col = "timestep_before_stopping",
        y_col = "abs_delta_action_aligned_action_logit",
        xlab = "Steps relative\nto stopping",
        ylab = "|delta action-aligned\nterminal logit|",
        y_lim = nonnegative_axis_limit(delta_action_logit_summary, "abs_delta_action_aligned_action_logit"),
        x_lim = relative_timestep_range(delta_action_logit_summary$timestep_before_stopping),
        hline = 0,
        vline = NULL
      )
    )
  )
}

if (nrow(delta_z_mu_summary) > 0L) {
  analysis_specs <- c(
    analysis_specs,
    list(
      list(
        name = "Abs delta z_mu by timestep",
        slug = "abs_delta_z_mu_by_timestep",
        summary = delta_z_mu_summary,
        x_col = "timestep_before_stopping",
        y_col = "abs_delta_z_mu",
        xlab = "Steps relative\nto stopping",
        ylab = "|delta prior-norm\nz_mu|",
        y_lim = positive_log_axis_limit(delta_z_mu_summary, "abs_delta_z_mu"),
        x_lim = relative_timestep_range(delta_z_mu_summary$timestep_before_stopping),
        hline = 0,
        vline = NULL,
        y_scale = "log10"
      )
    )
  )
}

if (nrow(delta_z_sigma_summary) > 0L) {
  analysis_specs <- c(
    analysis_specs,
    list(
      list(
        name = "Signed delta z_sigma by timestep",
        slug = "signed_delta_z_sigma_by_timestep",
        summary = delta_z_sigma_summary,
        x_col = "timestep_before_stopping",
        y_col = "signed_delta_z_sigma",
        xlab = "Steps relative\nto stopping",
        ylab = "Signed delta\nprior-norm z_sigma",
        y_lim = signed_log_axis_limit(delta_z_sigma_summary, "signed_delta_z_sigma"),
        x_lim = relative_timestep_range(delta_z_sigma_summary$timestep_before_stopping),
        hline = 0,
        vline = NULL,
        y_scale = "signed_log10"
      )
    )
  )
}

if (nrow(z_sigma_timestep_summary) > 0L) {
  analysis_specs <- c(
    analysis_specs,
    list(
      list(
        name = "Average z_sigma by timestep",
        slug = "average_z_sigma_by_timestep",
        summary = z_sigma_timestep_summary,
        x_col = "timestep_before_stopping",
        y_col = "average_z_sigma",
        xlab = "Steps relative\nto stopping",
        ylab = "Average prior-norm\nz_sigma",
        y_lim = positive_log_axis_limit(z_sigma_timestep_summary, "average_z_sigma"),
        x_lim = relative_timestep_range(z_sigma_timestep_summary$timestep_before_stopping),
        hline = NULL,
        vline = NULL,
        y_scale = "log10"
      )
    )
  )
}

append_response_locked_specs <- function(specs, response_summary) {
  if (nrow(response_summary) == 0L || !"metric" %in% names(response_summary)) {
    return(specs)
  }
  response_specs <- list(
    choice_aligned_cumulative_evidence = list(
      name = "Response-locked choice-aligned cumulative evidence",
      slug = "response_locked_choice_aligned_cumulative_evidence",
      ylab = "Choice-aligned\ncumulative evidence",
      y_lim = NULL,
      hline = 0
    ),
    p_stop = list(
      name = "Response-locked stop probability",
      slug = "response_locked_stopping_probability",
      ylab = "Probability\nof stopping",
      y_lim = c(0, 1),
      hline = 0.5
    ),
    p_eventual_choice_given_stop = list(
      name = "Response-locked eventual choice given stop",
      slug = "response_locked_eventual_choice_given_stop",
      ylab = "P(eventual choice)\nconditional on stop",
      y_lim = c(0, 1),
      hline = 0.5
    )
  )
  for (metric_name in names(response_specs)) {
    metric_summary <- response_summary[response_summary$metric == metric_name, , drop = FALSE]
    if (nrow(metric_summary) == 0L) {
      message(sprintf("Skipping response-locked %s: no rows after filtering/min-samples.", metric_name))
      next
    }
    spec <- response_specs[[metric_name]]
    specs <- c(
      specs,
      list(
        list(
          name = spec$name,
          slug = spec$slug,
          summary = metric_summary,
          x_col = "relative_timestep",
          y_col = "value",
          xlab = "Steps relative\nto stopping",
          ylab = spec$ylab,
          y_lim = spec$y_lim,
          x_lim = c(-max_steps_before_stop, 0),
          hline = spec$hline,
          vline = 0
        )
      )
    )
  }
  specs
}

analysis_specs <- append_response_locked_specs(analysis_specs, response_locked_summary)

append_response_locked_coherence_specs <- function(specs, response_summary) {
  if (
    nrow(response_summary) == 0L ||
      length(delta_coherence_values) == 0L ||
      !"coherence" %in% names(response_summary) ||
      !"metric" %in% names(response_summary)
  ) {
    return(specs)
  }
  metric_summary <- response_summary[
    response_summary$metric == "choice_aligned_cumulative_evidence",
    ,
    drop = FALSE
  ]
  if (nrow(metric_summary) == 0L) {
    message("Skipping coherence-specific response-locked cumulative evidence: no rows after filtering/min-samples.")
    return(specs)
  }
  for (coherence_value in delta_coherence_values) {
    coherence_summary <- metric_summary[parameter_equal(metric_summary$coherence, coherence_value), , drop = FALSE]
    if (nrow(coherence_summary) == 0L) {
      message(sprintf(
        "Skipping response-locked choice-aligned cumulative evidence coherence=%s: no rows after filtering/min-samples.",
        num_label(coherence_value)
      ))
      next
    }
    specs <- c(
      specs,
      list(
        list(
          name = sprintf(
            "Response-locked choice-aligned cumulative evidence, coherence=%s",
            num_label(coherence_value)
          ),
          slug = sprintf(
            "response_locked_choice_aligned_cumulative_evidence_coherence_%s",
            value_token(coherence_value)
          ),
          summary = coherence_summary,
          x_col = "relative_timestep",
          y_col = "value",
          xlab = "Steps relative\nto stopping",
          ylab = "Choice-aligned\ncumulative evidence",
          y_lim = NULL,
          x_lim = c(-max_steps_before_stop, 0),
          hline = 0,
          vline = 0
        )
      )
    )
  }
  specs
}

message("Coherence-specific response-locked cumulative-evidence files are disabled; writing coherence-overlay files instead.")

max_observation_limit <- suppressWarnings(max(as_num(trial_data$max_observations), na.rm = TRUE))
if (!is.finite(max_observation_limit)) {
  max_observation_limit <- max_steps_before_stop
}
stop_duration_xlim <- c(1, max(1, max_observation_limit))

if (nrow(threshold_by_duration_summary) > 0L) {
  analysis_specs <- c(
    analysis_specs,
    list(
      list(
        name = sprintf(
          "Decision threshold by stopping duration, coherence=%s",
          num_label(response_locked_coherence)
        ),
        slug = "decision_threshold_by_stopping_timestep",
        summary = threshold_by_duration_summary,
        x_col = "stopping_timestep",
        y_col = "choice_aligned_cumulative_evidence",
        xlab = "Total observations\nbefore stopping",
        ylab = "Choice-aligned cumulative\nevidence at stop",
        y_lim = NULL,
        x_lim = stop_duration_xlim,
        hline = 0,
        vline = NULL
      )
    )
  )
} else if (!observer_only_mode) {
  message("Skipping decision-threshold-by-duration plot: no rows after filtering/min-samples.")
}

append_threshold_by_coherence_specs <- function(specs, threshold_summary) {
  if (
    nrow(threshold_summary) == 0L ||
      length(delta_coherence_values) == 0L ||
      !"coherence" %in% names(threshold_summary)
  ) {
    return(specs)
  }
  for (coherence_value in delta_coherence_values) {
    coherence_summary <- threshold_summary[parameter_equal(threshold_summary$coherence, coherence_value), , drop = FALSE]
    if (nrow(coherence_summary) == 0L) {
      message(sprintf(
        "Skipping decision-threshold-by-duration coherence=%s: no rows after filtering/min-samples.",
        num_label(coherence_value)
      ))
      next
    }
    specs <- c(
      specs,
      list(
        list(
          name = sprintf(
            "Decision threshold by stopping duration, coherence=%s",
            num_label(coherence_value)
          ),
          slug = sprintf(
            "decision_threshold_by_stopping_timestep_coherence_%s",
            value_token(coherence_value)
          ),
          summary = coherence_summary,
          x_col = "stopping_timestep",
          y_col = "choice_aligned_cumulative_evidence",
          xlab = "Total observations\nbefore stopping",
          ylab = "Choice-aligned cumulative\nevidence at stop",
          y_lim = NULL,
          x_lim = stop_duration_xlim,
          hline = 0,
          vline = NULL
        )
      )
    )
  }
  specs
}

analysis_specs <- append_threshold_by_coherence_specs(analysis_specs, threshold_by_duration_coherence_summary)

append_coherence_delta_specs <- function(specs, summary, base_slug, base_name, y_col, ylab,
                                         nonnegative = FALSE, y_scale = "linear") {
  if (nrow(summary) == 0L || length(delta_coherence_values) == 0L || !"coherence" %in% names(summary)) {
    return(specs)
  }
  for (coherence_value in delta_coherence_values) {
    coherence_summary <- summary[parameter_equal(summary$coherence, coherence_value), , drop = FALSE]
    if (nrow(coherence_summary) == 0L) {
      message(sprintf(
        "Skipping %s coherence=%s: no rows after filtering/min-samples.",
        base_slug,
        num_label(coherence_value)
      ))
      next
    }
    specs <- c(
      specs,
      list(
        list(
          name = sprintf("%s, coherence=%s", base_name, num_label(coherence_value)),
          slug = sprintf("%s_coherence_%s", base_slug, value_token(coherence_value)),
          summary = coherence_summary,
          x_col = "timestep_before_stopping",
          y_col = y_col,
          xlab = "Steps relative\nto stopping",
          ylab = ylab,
          y_lim = if (identical(y_scale, "log10")) {
            positive_log_axis_limit(coherence_summary, y_col)
          } else if (identical(y_scale, "signed_log10")) {
            signed_log_axis_limit(coherence_summary, y_col)
          } else if (isTRUE(nonnegative)) {
            nonnegative_axis_limit(coherence_summary, y_col)
          } else {
            NULL
          },
          x_lim = relative_timestep_range(coherence_summary$timestep_before_stopping),
          hline = 0,
          vline = NULL,
          y_scale = y_scale
        )
      )
    )
  }
  specs
}

analysis_specs <- append_coherence_delta_specs(
  analysis_specs,
  delta_action_logit_by_coherence_summary,
  "abs_delta_action_aligned_action_logit_by_timestep",
  "Abs delta action-aligned terminal logit by timestep",
  "abs_delta_action_aligned_action_logit",
  "|delta action-aligned\nterminal logit|",
  nonnegative = TRUE
)
analysis_specs <- append_coherence_delta_specs(
  analysis_specs,
  delta_z_mu_by_coherence_summary,
  "abs_delta_z_mu_by_timestep",
  "Abs delta z_mu by timestep",
  "abs_delta_z_mu",
  "|delta prior-norm\nz_mu|",
  nonnegative = TRUE,
  y_scale = "log10"
)
analysis_specs <- append_coherence_delta_specs(
  analysis_specs,
  delta_z_sigma_by_coherence_summary,
  "signed_delta_z_sigma_by_timestep",
  "Signed delta z_sigma by timestep",
  "signed_delta_z_sigma",
  "Signed delta\nprior-norm z_sigma",
  nonnegative = FALSE,
  y_scale = "signed_log10"
)
analysis_specs <- append_coherence_delta_specs(
  analysis_specs,
  z_sigma_timestep_by_coherence_summary,
  "average_z_sigma_by_timestep",
  "Average z_sigma by timestep",
  "average_z_sigma",
  "Average prior-norm\nz_sigma",
  nonnegative = TRUE,
  y_scale = "log10"
)

obsstd_levels <- sort(unique(as_num(selected_obsstd_values)))
obsstd_levels <- obsstd_levels[is.finite(obsstd_levels)]

plot_file_for_spec <- function(spec) {
  sprintf("%s_%s.png", output_file_stem, spec$slug)
}

draw_coherence_header <- function(header, coherence_values, colors) {
  plot.new()
  text(0.5, 0.82, header, cex = 1)
  coherence_values <- sort(unique(as_num(coherence_values)))
  coherence_values <- coherence_values[is.finite(coherence_values)]
  if (length(coherence_values) == 0L) return(invisible(NULL))
  labels <- paste0("coh ", vapply(coherence_values, num_label, character(1)))
  graphics::legend(
    "bottom",
    legend = labels,
    title = "Coherence",
    col = colors[as.character(coherence_values)],
    pch = 16,
    lwd = 1.3,
    bty = "n",
    cex = 0.82,
    pt.cex = 0.8,
    horiz = TRUE,
    x.intersp = 0.7,
    y.intersp = 0.85
  )
  invisible(NULL)
}

plot_coherence_overlay_panel <- function(summary_data, family, param, x_lim, y_lim,
                                         coherence_values, coherence_colors,
                                         xlab, ylab, main = "",
                                         x_col = "relative_timestep",
                                         y_col = "value",
                                         sem_col = NULL,
                                         vline = 0,
                                         hline = 0) {
  panel_data <- summary_data[
    summary_data$family == family &
      parameter_equal(summary_data$parameter_value, param),
    ,
    drop = FALSE
  ]
  plot(
    NA,
    xlim = x_lim,
    ylim = y_lim,
    xlab = xlab,
    ylab = ylab,
    main = main,
    xaxt = "n",
    cex.lab = 1,
    cex.axis = 1,
    cex.main = 1
  )
  axis_ticks <- seq.int(ceiling(x_lim[[1L]]), floor(x_lim[[2L]]), by = 1L)
  if (length(axis_ticks) > 0L) axis(1, at = axis_ticks)
  grid(col = "grey90")
  if (!is.null(hline)) {
    abline(h = hline, col = "grey55", lty = 2, lwd = 0.8)
  }
  if (!is.null(vline)) {
    abline(v = vline, col = "grey55", lty = 2, lwd = 0.8)
  }
  if (!x_col %in% names(panel_data) || !y_col %in% names(panel_data)) {
    box()
    return(invisible(NULL))
  }
  if (is.null(sem_col)) {
    sem_col <- paste0(y_col, "_sem")
  }
  for (coherence_value in coherence_values) {
    line_data <- panel_data[parameter_equal(panel_data$coherence, coherence_value), , drop = FALSE]
    if (nrow(line_data) == 0L) next
    line_data <- line_data[order(as_num(line_data[[x_col]])), , drop = FALSE]
    x <- as_num(line_data[[x_col]])
    y <- as_num(line_data[[y_col]])
    keep <- is.finite(x) & is.finite(y)
    if (!any(keep)) next
    x <- x[keep]
    y <- y[keep]
    col <- coherence_colors[[as.character(coherence_value)]]
    lines(x, y, col = col, lwd = 1.3)
    points(x, y, col = col, pch = 16, cex = 0.65)
    if (sem_col %in% names(line_data)) {
      draw_error_bars(x, y, line_data[[sem_col]][keep], col)
    }
  }
  box()
  invisible(NULL)
}

save_response_locked_coherence_overlay_plots <- function() {
  if (nrow(response_locked_by_coherence_summary) == 0L ||
      length(delta_coherence_values) == 0L ||
      !"coherence" %in% names(response_locked_by_coherence_summary) ||
      !"metric" %in% names(response_locked_by_coherence_summary)) {
    message("Skipping coherence-overlay response-locked cumulative evidence: missing rows or coherence column.")
    return(invisible(NULL))
  }
  metric_summary <- response_locked_by_coherence_summary[
    response_locked_by_coherence_summary$metric == "choice_aligned_cumulative_evidence",
    ,
    drop = FALSE
  ]
  if (nrow(metric_summary) == 0L) {
    message("Skipping coherence-overlay response-locked cumulative evidence: no cumulative-evidence rows.")
    return(invisible(NULL))
  }
  coherence_values <- sort(unique(as_num(delta_coherence_values)))
  coherence_values <- coherence_values[is.finite(coherence_values)]
  coherence_values <- coherence_values[
    vapply(coherence_values, function(v) any(parameter_equal(metric_summary$coherence, v)), logical(1))
  ]
  if (length(coherence_values) == 0L) {
    message("Skipping coherence-overlay response-locked cumulative evidence: requested coherence values are absent.")
    return(invisible(NULL))
  }
  coherence_colors <- coherence_color_values(coherence_values)
  beta_params <- sort(unique(as_num(beta_values)))
  beta_params <- beta_params[is.finite(beta_params)]
  opp_params <- sort(unique(as_num(opportunity_values)))
  opp_params <- opp_params[is.finite(opp_params)]
  n_param_rows <- max(length(beta_params), length(opp_params))
  if (n_param_rows == 0L) {
    return(invisible(NULL))
  }
  shared_x_lim <- c(-max_steps_before_stop, 0)
  shared_y_lim <- shared_axis_limit(metric_summary, "value", include_sem = TRUE, fallback = c(0, 1))

  for (obsstd in obsstd_levels) {
    obs_data <- metric_summary[parameter_equal(metric_summary$observation_noise_std, obsstd), , drop = FALSE]
    if (nrow(obs_data) == 0L) {
      message(sprintf(
        "Skipping coherence-overlay response-locked cumulative evidence for obsstd=%s: no rows.",
        num_label(obsstd)
      ))
      next
    }
    output_file <- sprintf(
      "%s_response_locked_choice_aligned_cumulative_evidence_coherence_overlay_obsstd_%s.png",
      output_file_stem,
      value_token(obsstd)
    )
    header_row_height_in <- 0.95
    panel_cell_width_in <- target_panel_side_in + panel_margin_in[["left"]] + panel_margin_in[["right"]]
    panel_cell_height_in <- target_panel_side_in + panel_margin_in[["bottom"]] + panel_margin_in[["top"]]
    device_width_in <- 2 * panel_cell_width_in
    device_height_in <- header_row_height_in + n_param_rows * panel_cell_height_in
    grDevices::png(
      output_file,
      width = device_width_in,
      height = device_height_in,
      units = "in",
      res = 300,
      pointsize = 7
    )
    layout(
      matrix(seq_len((n_param_rows + 1L) * 2L), nrow = n_param_rows + 1L, ncol = 2L, byrow = TRUE),
      widths = c(1, 1),
      heights = c(header_row_height_in / panel_cell_height_in, rep(1, n_param_rows))
    )
    old_par <- par(no.readonly = TRUE)
    on.exit({
      invisible(try(par(old_par), silent = TRUE))
      grDevices::dev.off()
    }, add = TRUE)
    par(cex = 1, cex.axis = 1, cex.lab = 1, cex.main = 1, oma = c(0, 0, 0, 0), xpd = FALSE)
    par(mai = c(0.02, 0.02, 0.02, 0.08))
    draw_coherence_header(
      sprintf("std = %s\nVarying memory lambda", num_label(obsstd)),
      coherence_values,
      coherence_colors
    )
    par(mai = c(0.02, 0.02, 0.02, 0.08))
    draw_coherence_header(
      sprintf("std = %s\nVarying opportunity cost", num_label(obsstd)),
      coherence_values,
      coherence_colors
    )
    for (row_i in seq_len(n_param_rows)) {
      par(mai = panel_margin_in)
      if (row_i <= length(beta_params)) {
        plot_coherence_overlay_panel(
          obs_data,
          "beta",
          beta_params[[row_i]],
          shared_x_lim,
          shared_y_lim,
          coherence_values,
          coherence_colors,
          xlab = "Steps relative\nto stopping",
          ylab = "Choice-aligned\ncumulative evidence",
          main = sprintf("lambda = %s", num_label(beta_params[[row_i]]))
        )
      } else {
        plot.new()
      }
      par(mai = panel_margin_in)
      if (row_i <= length(opp_params)) {
        plot_coherence_overlay_panel(
          obs_data,
          "opportunity",
          opp_params[[row_i]],
          shared_x_lim,
          shared_y_lim,
          coherence_values,
          coherence_colors,
          xlab = "Steps relative\nto stopping",
          ylab = "Choice-aligned\ncumulative evidence",
          main = sprintf("opp = %s", num_label(opp_params[[row_i]]))
        )
      } else {
        plot.new()
      }
    }
    invisible(try(par(old_par), silent = TRUE))
    grDevices::dev.off()
    on.exit(NULL, add = FALSE)
    message(sprintf("Saved %s", output_file))
  }
  invisible(NULL)
}

save_response_locked_target_probability_overlay_plots <- function() {
  if (observer_only_mode) {
    message("Skipping response-locked target-choice probability overlay for observer-only runs.")
    return(invisible(NULL))
  }
  if (nrow(full_policy_target_response_summary) == 0L ||
      length(delta_coherence_values) == 0L ||
      !"coherence" %in% names(full_policy_target_response_summary)) {
    message(
      "Skipping response-locked target-choice probability overlay: missing full_policy_* rows. ",
      "Regenerate evidence simulation CSVs with the current simulator so full_policy_choose_a/b_t* columns are written."
    )
    return(invisible(NULL))
  }
  metric_summary <- full_policy_target_response_summary
  coherence_values <- sort(unique(as_num(delta_coherence_values)))
  coherence_values <- coherence_values[is.finite(coherence_values)]
  coherence_values <- coherence_values[
    vapply(coherence_values, function(v) any(parameter_equal(metric_summary$coherence, v)), logical(1))
  ]
  if (length(coherence_values) == 0L) {
    message("Skipping response-locked target-choice probability overlay: requested coherence values are absent.")
    return(invisible(NULL))
  }
  coherence_colors <- coherence_color_values(coherence_values)
  beta_params <- sort(unique(as_num(beta_values)))
  beta_params <- beta_params[is.finite(beta_params)]
  opp_params <- sort(unique(as_num(opportunity_values)))
  opp_params <- opp_params[is.finite(opp_params)]
  n_param_rows <- max(length(beta_params), length(opp_params))
  if (n_param_rows == 0L) {
    return(invisible(NULL))
  }
  x_values <- as_num(metric_summary$relative_timestep)
  x_values <- x_values[is.finite(x_values)]
  if (length(x_values) == 0L) {
    return(invisible(NULL))
  }
  shared_x_lim <- range(c(x_values, 0), finite = TRUE)
  shared_x_lim <- c(floor(shared_x_lim[[1L]]), ceiling(shared_x_lim[[2L]]))
  shared_y_lim <- c(0, 1)

  for (obsstd in obsstd_levels) {
    obs_data <- metric_summary[parameter_equal(metric_summary$observation_noise_std, obsstd), , drop = FALSE]
    if (nrow(obs_data) == 0L) {
      message(sprintf(
        "Skipping response-locked target-choice probability overlay for obsstd=%s: no rows.",
        num_label(obsstd)
      ))
      next
    }
    output_file <- sprintf(
      "%s_response_locked_target_choice_probability_coherence_overlay_obsstd_%s.png",
      output_file_stem,
      value_token(obsstd)
    )
    header_row_height_in <- 0.95
    panel_cell_width_in <- target_panel_side_in + panel_margin_in[["left"]] + panel_margin_in[["right"]]
    panel_cell_height_in <- target_panel_side_in + panel_margin_in[["bottom"]] + panel_margin_in[["top"]]
    device_width_in <- 2 * panel_cell_width_in
    device_height_in <- header_row_height_in + n_param_rows * panel_cell_height_in
    grDevices::png(
      output_file,
      width = device_width_in,
      height = device_height_in,
      units = "in",
      res = 300,
      pointsize = 7
    )
    layout(
      matrix(seq_len((n_param_rows + 1L) * 2L), nrow = n_param_rows + 1L, ncol = 2L, byrow = TRUE),
      widths = c(1, 1),
      heights = c(header_row_height_in / panel_cell_height_in, rep(1, n_param_rows))
    )
    old_par <- par(no.readonly = TRUE)
    on.exit({
      invisible(try(par(old_par), silent = TRUE))
      grDevices::dev.off()
    }, add = TRUE)
    par(cex = 1, cex.axis = 1, cex.lab = 1, cex.main = 1, oma = c(0, 0, 0, 0), xpd = FALSE)
    par(mai = c(0.02, 0.02, 0.02, 0.08))
    draw_coherence_header(
      sprintf("std = %s\nVarying memory lambda", num_label(obsstd)),
      coherence_values,
      coherence_colors
    )
    par(mai = c(0.02, 0.02, 0.02, 0.08))
    draw_coherence_header(
      sprintf("std = %s\nVarying opportunity cost", num_label(obsstd)),
      coherence_values,
      coherence_colors
    )
    for (row_i in seq_len(n_param_rows)) {
      par(mai = panel_margin_in)
      if (row_i <= length(beta_params)) {
        plot_coherence_overlay_panel(
          obs_data,
          "beta",
          beta_params[[row_i]],
          shared_x_lim,
          shared_y_lim,
          coherence_values,
          coherence_colors,
          xlab = "Steps relative\nto stopping",
          ylab = "P(target | choose)",
          main = sprintf("lambda = %s", num_label(beta_params[[row_i]])),
          y_col = "p_target_given_choice",
          hline = 0.5
        )
      } else {
        plot.new()
      }
      par(mai = panel_margin_in)
      if (row_i <= length(opp_params)) {
        plot_coherence_overlay_panel(
          obs_data,
          "opportunity",
          opp_params[[row_i]],
          shared_x_lim,
          shared_y_lim,
          coherence_values,
          coherence_colors,
          xlab = "Steps relative\nto stopping",
          ylab = "P(target | choose)",
          main = sprintf("opp = %s", num_label(opp_params[[row_i]])),
          y_col = "p_target_given_choice",
          hline = 0.5
        )
      } else {
        plot.new()
      }
    }
    invisible(try(par(old_par), silent = TRUE))
    grDevices::dev.off()
    on.exit(NULL, add = FALSE)
    message(sprintf("Saved %s", output_file))
  }
  invisible(NULL)
}

save_chronological_cumulative_evidence_coherence_overlay_plots <- function() {
  if (nrow(chronological_by_coherence_summary) == 0L ||
      length(delta_coherence_values) == 0L ||
      !"coherence" %in% names(chronological_by_coherence_summary) ||
      !"metric" %in% names(chronological_by_coherence_summary)) {
    message("Skipping chronological coherence-overlay cumulative evidence: missing rows or coherence column.")
    return(invisible(NULL))
  }
  metric_summary <- chronological_by_coherence_summary[
    chronological_by_coherence_summary$metric == "choice_aligned_cumulative_evidence",
    ,
    drop = FALSE
  ]
  if (nrow(metric_summary) == 0L) {
    message("Skipping chronological coherence-overlay cumulative evidence: no cumulative-evidence rows.")
    return(invisible(NULL))
  }
  coherence_values <- sort(unique(as_num(delta_coherence_values)))
  coherence_values <- coherence_values[is.finite(coherence_values)]
  coherence_values <- coherence_values[
    vapply(coherence_values, function(v) any(parameter_equal(metric_summary$coherence, v)), logical(1))
  ]
  if (length(coherence_values) == 0L) {
    message("Skipping chronological coherence-overlay cumulative evidence: requested coherence values are absent.")
    return(invisible(NULL))
  }
  coherence_colors <- coherence_color_values(coherence_values)
  beta_params <- sort(unique(as_num(beta_values)))
  beta_params <- beta_params[is.finite(beta_params)]
  opp_params <- sort(unique(as_num(opportunity_values)))
  opp_params <- opp_params[is.finite(opp_params)]
  n_param_rows <- max(length(beta_params), length(opp_params))
  if (n_param_rows == 0L) {
    return(invisible(NULL))
  }
  timestep_lim <- c(1, max(1, max_observation_limit))
  shared_y_lim <- shared_axis_limit(metric_summary, "value", include_sem = TRUE, fallback = c(0, 1))

  for (obsstd in obsstd_levels) {
    obs_data <- metric_summary[parameter_equal(metric_summary$observation_noise_std, obsstd), , drop = FALSE]
    if (nrow(obs_data) == 0L) {
      message(sprintf(
        "Skipping chronological coherence-overlay cumulative evidence for obsstd=%s: no rows.",
        num_label(obsstd)
      ))
      next
    }
    output_file <- sprintf(
      "%s_choice_aligned_cumulative_evidence_by_timestep_coherence_overlay_obsstd_%s.png",
      output_file_stem,
      value_token(obsstd)
    )
    header_row_height_in <- 0.95
    panel_cell_width_in <- target_panel_side_in + panel_margin_in[["left"]] + panel_margin_in[["right"]]
    panel_cell_height_in <- target_panel_side_in + panel_margin_in[["bottom"]] + panel_margin_in[["top"]]
    device_width_in <- 2 * panel_cell_width_in
    device_height_in <- header_row_height_in + n_param_rows * panel_cell_height_in
    grDevices::png(
      output_file,
      width = device_width_in,
      height = device_height_in,
      units = "in",
      res = 300,
      pointsize = 7
    )
    layout(
      matrix(seq_len((n_param_rows + 1L) * 2L), nrow = n_param_rows + 1L, ncol = 2L, byrow = TRUE),
      widths = c(1, 1),
      heights = c(header_row_height_in / panel_cell_height_in, rep(1, n_param_rows))
    )
    old_par <- par(no.readonly = TRUE)
    on.exit({
      invisible(try(par(old_par), silent = TRUE))
      grDevices::dev.off()
    }, add = TRUE)
    par(cex = 1, cex.axis = 1, cex.lab = 1, cex.main = 1, oma = c(0, 0, 0, 0), xpd = FALSE)
    par(mai = c(0.02, 0.02, 0.02, 0.08))
    draw_coherence_header(
      sprintf("std = %s\nVarying memory lambda", num_label(obsstd)),
      coherence_values,
      coherence_colors
    )
    par(mai = c(0.02, 0.02, 0.02, 0.08))
    draw_coherence_header(
      sprintf("std = %s\nVarying opportunity cost", num_label(obsstd)),
      coherence_values,
      coherence_colors
    )
    for (row_i in seq_len(n_param_rows)) {
      par(mai = panel_margin_in)
      if (row_i <= length(beta_params)) {
        plot_coherence_overlay_panel(
          obs_data,
          "beta",
          beta_params[[row_i]],
          timestep_lim,
          shared_y_lim,
          coherence_values,
          coherence_colors,
          xlab = "Observation\ntimestep",
          ylab = "Choice-aligned\ncumulative evidence",
          main = sprintf("lambda = %s", num_label(beta_params[[row_i]])),
          x_col = "timestep",
          vline = NULL
        )
      } else {
        plot.new()
      }
      par(mai = panel_margin_in)
      if (row_i <= length(opp_params)) {
        plot_coherence_overlay_panel(
          obs_data,
          "opportunity",
          opp_params[[row_i]],
          timestep_lim,
          shared_y_lim,
          coherence_values,
          coherence_colors,
          xlab = "Observation\ntimestep",
          ylab = "Choice-aligned\ncumulative evidence",
          main = sprintf("opp = %s", num_label(opp_params[[row_i]])),
          x_col = "timestep",
          vline = NULL
        )
      } else {
        plot.new()
      }
    }
    invisible(try(par(old_par), silent = TRUE))
    grDevices::dev.off()
    on.exit(NULL, add = FALSE)
    message(sprintf("Saved %s", output_file))
  }
  invisible(NULL)
}

plot_threshold_overlay_panel <- function(summary_data, family, param, x_lim, y_lim,
                                         coherence_values, coherence_colors,
                                         xlab, ylab, main = "") {
  panel_data <- summary_data[
    summary_data$family == family &
      parameter_equal(summary_data$parameter_value, param),
    ,
    drop = FALSE
  ]
  plot(
    NA,
    xlim = x_lim,
    ylim = y_lim,
    xlab = xlab,
    ylab = ylab,
    main = main,
    xaxt = "n",
    cex.lab = 1,
    cex.axis = 1,
    cex.main = 1
  )
  axis_ticks <- seq.int(ceiling(x_lim[[1L]]), floor(x_lim[[2L]]), by = 1L)
  if (length(axis_ticks) > 0L) axis(1, at = axis_ticks)
  grid(col = "grey90")
  abline(h = 0, col = "grey55", lty = 2, lwd = 0.8)
  for (coherence_value in coherence_values) {
    line_data <- panel_data[parameter_equal(panel_data$coherence, coherence_value), , drop = FALSE]
    if (nrow(line_data) == 0L) next
    line_data <- line_data[order(as_num(line_data$stopping_timestep)), , drop = FALSE]
    x <- as_num(line_data$stopping_timestep)
    y <- as_num(line_data$choice_aligned_cumulative_evidence)
    col <- coherence_colors[[as.character(coherence_value)]]
    lines(x, y, col = col, lwd = 1.3)
    points(x, y, col = col, pch = 16, cex = 0.65)
    if ("choice_aligned_cumulative_evidence_sem" %in% names(line_data)) {
      draw_error_bars(x, y, line_data$choice_aligned_cumulative_evidence_sem, col)
    }
  }
  box()
}

save_decision_threshold_coherence_overlay_plots <- function() {
  if (observer_only_mode) {
    return(invisible(NULL))
  }
  if (nrow(threshold_by_duration_coherence_summary) == 0L ||
      length(delta_coherence_values) == 0L ||
      !"coherence" %in% names(threshold_by_duration_coherence_summary)) {
    message("Skipping coherence-overlay decision threshold plot: missing rows or coherence column.")
    return(invisible(NULL))
  }
  threshold_summary <- threshold_by_duration_coherence_summary
  coherence_values <- sort(unique(as_num(delta_coherence_values)))
  coherence_values <- coherence_values[is.finite(coherence_values)]
  coherence_values <- coherence_values[
    vapply(coherence_values, function(v) any(parameter_equal(threshold_summary$coherence, v)), logical(1))
  ]
  if (length(coherence_values) == 0L) {
    message("Skipping coherence-overlay decision threshold plot: requested coherence values are absent.")
    return(invisible(NULL))
  }
  coherence_colors <- coherence_color_values(coherence_values)
  beta_params <- sort(unique(as_num(beta_values)))
  beta_params <- beta_params[is.finite(beta_params)]
  opp_params <- sort(unique(as_num(opportunity_values)))
  opp_params <- opp_params[is.finite(opp_params)]
  n_param_rows <- max(length(beta_params), length(opp_params))
  if (n_param_rows == 0L) {
    return(invisible(NULL))
  }
  shared_y_lim <- shared_axis_limit(
    threshold_summary,
    "choice_aligned_cumulative_evidence",
    include_sem = TRUE,
    fallback = c(0, 1)
  )

  for (obsstd in obsstd_levels) {
    obs_data <- threshold_summary[parameter_equal(threshold_summary$observation_noise_std, obsstd), , drop = FALSE]
    if (nrow(obs_data) == 0L) {
      message(sprintf(
        "Skipping coherence-overlay decision threshold for obsstd=%s: no rows.",
        num_label(obsstd)
      ))
      next
    }
    output_file <- sprintf(
      "%s_decision_threshold_by_stopping_timestep_coherence_overlay_obsstd_%s.png",
      output_file_stem,
      value_token(obsstd)
    )
    header_row_height_in <- 0.95
    panel_cell_width_in <- target_panel_side_in + panel_margin_in[["left"]] + panel_margin_in[["right"]]
    panel_cell_height_in <- target_panel_side_in + panel_margin_in[["bottom"]] + panel_margin_in[["top"]]
    device_width_in <- 2 * panel_cell_width_in
    device_height_in <- header_row_height_in + n_param_rows * panel_cell_height_in
    grDevices::png(
      output_file,
      width = device_width_in,
      height = device_height_in,
      units = "in",
      res = 300,
      pointsize = 7
    )
    layout(
      matrix(seq_len((n_param_rows + 1L) * 2L), nrow = n_param_rows + 1L, ncol = 2L, byrow = TRUE),
      widths = c(1, 1),
      heights = c(header_row_height_in / panel_cell_height_in, rep(1, n_param_rows))
    )
    old_par <- par(no.readonly = TRUE)
    on.exit({
      invisible(try(par(old_par), silent = TRUE))
      grDevices::dev.off()
    }, add = TRUE)
    par(cex = 1, cex.axis = 1, cex.lab = 1, cex.main = 1, oma = c(0, 0, 0, 0), xpd = FALSE)
    par(mai = c(0.02, 0.02, 0.02, 0.08))
    draw_coherence_header(
      sprintf("std = %s\nVarying memory lambda", num_label(obsstd)),
      coherence_values,
      coherence_colors
    )
    par(mai = c(0.02, 0.02, 0.02, 0.08))
    draw_coherence_header(
      sprintf("std = %s\nVarying opportunity cost", num_label(obsstd)),
      coherence_values,
      coherence_colors
    )
    for (row_i in seq_len(n_param_rows)) {
      par(mai = panel_margin_in)
      if (row_i <= length(beta_params)) {
        plot_threshold_overlay_panel(
          obs_data,
          "beta",
          beta_params[[row_i]],
          stop_duration_xlim,
          shared_y_lim,
          coherence_values,
          coherence_colors,
          xlab = "Total observations\nbefore stopping",
          ylab = "Choice-aligned cumulative\nevidence at stop",
          main = sprintf("lambda = %s", num_label(beta_params[[row_i]]))
        )
      } else {
        plot.new()
      }
      par(mai = panel_margin_in)
      if (row_i <= length(opp_params)) {
        plot_threshold_overlay_panel(
          obs_data,
          "opportunity",
          opp_params[[row_i]],
          stop_duration_xlim,
          shared_y_lim,
          coherence_values,
          coherence_colors,
          xlab = "Total observations\nbefore stopping",
          ylab = "Choice-aligned cumulative\nevidence at stop",
          main = sprintf("opp = %s", num_label(opp_params[[row_i]]))
        )
      } else {
        plot.new()
      }
    }
    invisible(try(par(old_par), silent = TRUE))
    grDevices::dev.off()
    on.exit(NULL, add = FALSE)
    message(sprintf("Saved %s", output_file))
  }
  invisible(NULL)
}

save_analysis_plot <- function(spec) {
  spec_y_scale <- if (!is.null(spec$y_scale)) spec$y_scale else "linear"
  spec_y_floor <- y_scale_floor(spec$summary, spec$y_col, spec_y_scale)
  n_plot_rows <- length(obsstd_levels)
  shared_x_lim <- shared_axis_limit(spec$summary, spec$x_col, fixed_lim = spec$x_lim, fallback = c(0, 1))
  shared_y_lim <- shared_axis_limit(spec$summary, spec$y_col, fixed_lim = spec$y_lim, include_sem = TRUE, fallback = c(0, 1))
  next_layout_id <- 1L
  legend_beta_id <- next_layout_id
  next_layout_id <- next_layout_id + 1L
  legend_opp_id <- next_layout_id
  next_layout_id <- next_layout_id + 1L
  layout_matrix <- matrix(0L, nrow = n_plot_rows + 1L, ncol = 3L)
  layout_matrix[1L, ] <- c(0L, legend_beta_id, legend_opp_id)
  std_label_ids <- integer(n_plot_rows)
  beta_panel_ids <- integer(n_plot_rows)
  opp_panel_ids <- integer(n_plot_rows)
  for (row_i in seq_len(n_plot_rows)) {
    std_label_ids[[row_i]] <- next_layout_id
    next_layout_id <- next_layout_id + 1L
    beta_panel_ids[[row_i]] <- next_layout_id
    next_layout_id <- next_layout_id + 1L
    opp_panel_ids[[row_i]] <- next_layout_id
    next_layout_id <- next_layout_id + 1L
    layout_matrix[row_i + 1L, ] <- c(std_label_ids[[row_i]], beta_panel_ids[[row_i]], opp_panel_ids[[row_i]])
  }
  output_file <- plot_file_for_spec(spec)
  label_col_width_in <- 0.42
  header_row_height_in <- 0.9
  panel_cell_width_in <- target_panel_side_in + panel_margin_in[["left"]] + panel_margin_in[["right"]]
  panel_cell_height_in <- target_panel_side_in + panel_margin_in[["bottom"]] + panel_margin_in[["top"]]
  device_width_in <- label_col_width_in + 2 * panel_cell_width_in
  device_height_in <- header_row_height_in + n_plot_rows * panel_cell_height_in
  grDevices::png(
    output_file,
    width = device_width_in,
    height = device_height_in,
    units = "in",
    res = 300,
    pointsize = 7
  )
  layout(
    layout_matrix,
    widths = c(label_col_width_in / panel_cell_width_in, 1, 1),
    heights = c(header_row_height_in / panel_cell_height_in, rep(1, n_plot_rows))
  )
  old_par <- par(no.readonly = TRUE)
  par(cex = 1, cex.axis = 1, cex.lab = 1, cex.main = 1, oma = c(0, 0, 0, 0), xpd = FALSE)
  par(mai = c(0.02, 0.02, 0.02, 0.08))
  draw_family_header(
    "beta",
    beta_values,
    beta_colors,
    sprintf("Varying memory lambda\nopp = %s", num_label(fixed_opp)),
    expression(lambda[KL])
  )
  par(mai = c(0.02, 0.02, 0.02, 0.08))
  draw_family_header(
    "opportunity",
    opportunity_values,
    opp_colors,
    sprintf("Varying opportunity cost\nmemory lambda = %s", num_label(fixed_memory_lambda)),
    "Opportunity cost"
  )

  for (row_i in seq_along(obsstd_levels)) {
    obsstd <- obsstd_levels[[row_i]]
    spec_data <- spec$summary[
      parameter_equal(spec$summary$observation_noise_std, obsstd),
      ,
      drop = FALSE
    ]
    par(mai = label_margin_in)
    draw_obsstd_label(obsstd)
    par(mai = panel_margin_in)
    plot_curve_panel(
      spec_data, "beta", spec$x_col, spec$y_col,
      spec$xlab, spec$ylab, "",
      beta_colors, beta_values,
      y_lim = shared_y_lim, x_lim = shared_x_lim, hline = spec$hline, vline = spec$vline,
      y_scale = spec_y_scale, y_floor = spec_y_floor
    )
    par(mai = panel_margin_in)
    plot_curve_panel(
      spec_data, "opportunity", spec$x_col, spec$y_col,
      spec$xlab, spec$ylab, "",
      opp_colors, opportunity_values,
      y_lim = shared_y_lim, x_lim = shared_x_lim, hline = spec$hline, vline = spec$vline,
      y_scale = spec_y_scale, y_floor = spec_y_floor
    )
  }
  invisible(try(par(old_par), silent = TRUE))
  grDevices::dev.off()
  message(sprintf("Saved %s", output_file))
  invisible(output_file)
}

save_simple_two_column_plot <- function(spec, row_values, row_col, row_label_fn, output_file,
                                        fixed_obsstd = simple_fixed_obsstd) {
  if (nrow(spec$summary) == 0L || length(row_values) == 0L || !row_col %in% names(spec$summary)) {
    message(sprintf("Skipping simple %s: missing rows or column %s.", spec$slug, row_col))
    return(invisible(NULL))
  }
  spec_y_scale <- if (!is.null(spec$y_scale)) spec$y_scale else "linear"
  fixed_data <- spec$summary[
    parameter_equal(spec$summary$observation_noise_std, fixed_obsstd),
    ,
    drop = FALSE
  ]
  if (nrow(fixed_data) == 0L) {
    message(sprintf(
      "Skipping simple %s: no rows for observation-noise std=%s.",
      spec$slug,
      num_label(fixed_obsstd)
    ))
    return(invisible(NULL))
  }
  row_values <- row_values[vapply(row_values, function(v) any(parameter_equal(fixed_data[[row_col]], v)), logical(1))]
  if (length(row_values) == 0L) {
    message(sprintf("Skipping simple %s: no requested row values are present.", spec$slug))
    return(invisible(NULL))
  }
  fixed_data <- fixed_data[vapply(fixed_data[[row_col]], function(v) any(parameter_equal(row_values, v)), logical(1)), , drop = FALSE]
  if (nrow(fixed_data) == 0L) {
    message(sprintf("Skipping simple %s: no rows after row-value filter.", spec$slug))
    return(invisible(NULL))
  }

  spec_y_floor <- y_scale_floor(fixed_data, spec$y_col, spec_y_scale)
  shared_x_lim <- shared_axis_limit(fixed_data, spec$x_col, fixed_lim = spec$x_lim, fallback = c(0, 1))
  shared_y_lim <- shared_axis_limit(fixed_data, spec$y_col, fixed_lim = spec$y_lim, include_sem = TRUE, fallback = c(0, 1))
  n_plot_rows <- length(row_values)
  next_layout_id <- 1L
  legend_beta_id <- next_layout_id
  next_layout_id <- next_layout_id + 1L
  legend_opp_id <- next_layout_id
  next_layout_id <- next_layout_id + 1L
  layout_matrix <- matrix(0L, nrow = n_plot_rows + 1L, ncol = 3L)
  layout_matrix[1L, ] <- c(0L, legend_beta_id, legend_opp_id)
  label_ids <- integer(n_plot_rows)
  beta_panel_ids <- integer(n_plot_rows)
  opp_panel_ids <- integer(n_plot_rows)
  for (row_i in seq_len(n_plot_rows)) {
    label_ids[[row_i]] <- next_layout_id
    next_layout_id <- next_layout_id + 1L
    beta_panel_ids[[row_i]] <- next_layout_id
    next_layout_id <- next_layout_id + 1L
    opp_panel_ids[[row_i]] <- next_layout_id
    next_layout_id <- next_layout_id + 1L
    layout_matrix[row_i + 1L, ] <- c(label_ids[[row_i]], beta_panel_ids[[row_i]], opp_panel_ids[[row_i]])
  }

  label_col_width_in <- 0.52
  header_row_height_in <- 0.9
  panel_cell_width_in <- target_panel_side_in + panel_margin_in[["left"]] + panel_margin_in[["right"]]
  panel_cell_height_in <- target_panel_side_in + panel_margin_in[["bottom"]] + panel_margin_in[["top"]]
  device_width_in <- label_col_width_in + 2 * panel_cell_width_in
  device_height_in <- header_row_height_in + n_plot_rows * panel_cell_height_in
  grDevices::png(
    output_file,
    width = device_width_in,
    height = device_height_in,
    units = "in",
    res = 300,
    pointsize = 7
  )
  layout(
    layout_matrix,
    widths = c(label_col_width_in / panel_cell_width_in, 1, 1),
    heights = c(header_row_height_in / panel_cell_height_in, rep(1, n_plot_rows))
  )
  old_par <- par(no.readonly = TRUE)
  on.exit({
    invisible(try(par(old_par), silent = TRUE))
    grDevices::dev.off()
  }, add = TRUE)
  par(cex = 1, cex.axis = 1, cex.lab = 1, cex.main = 1, oma = c(0, 0, 0, 0), xpd = FALSE)
  par(mai = c(0.02, 0.02, 0.02, 0.08))
  draw_family_header(
    "beta",
    beta_values,
    beta_colors,
    sprintf("Varying memory lambda\nopp = %s", num_label(fixed_opp)),
    expression(lambda[KL])
  )
  par(mai = c(0.02, 0.02, 0.02, 0.08))
  draw_family_header(
    "opportunity",
    opportunity_values,
    opp_colors,
    sprintf("Varying opportunity cost\nmemory lambda = %s", num_label(fixed_memory_lambda)),
    "Opportunity cost"
  )

  for (row_i in seq_along(row_values)) {
    row_value <- row_values[[row_i]]
    spec_data <- fixed_data[parameter_equal(fixed_data[[row_col]], row_value), , drop = FALSE]
    par(mai = label_margin_in)
    draw_side_label(row_label_fn(row_value))
    par(mai = panel_margin_in)
    plot_curve_panel(
      spec_data, "beta", spec$x_col, spec$y_col,
      spec$xlab, spec$ylab, "",
      beta_colors, beta_values,
      y_lim = shared_y_lim, x_lim = shared_x_lim, hline = spec$hline, vline = spec$vline,
      y_scale = spec_y_scale, y_floor = spec_y_floor
    )
    par(mai = panel_margin_in)
    plot_curve_panel(
      spec_data, "opportunity", spec$x_col, spec$y_col,
      spec$xlab, spec$ylab, "",
      opp_colors, opportunity_values,
      y_lim = shared_y_lim, x_lim = shared_x_lim, hline = spec$hline, vline = spec$vline,
      y_scale = spec_y_scale, y_floor = spec_y_floor
    )
  }
  message(sprintf("Saved %s", output_file))
  invisible(output_file)
}

save_simple_evidence_plots <- function() {
  if (!is.finite(simple_fixed_obsstd)) {
    return(invisible(NULL))
  }
  simple_dir_name <- if (!is.null(option_simple_output_subdir$value) && nzchar(trim_string(option_simple_output_subdir$value))) {
    trim_string(option_simple_output_subdir$value)
  } else if (observer_only_mode) {
    "observer_only_simple"
  } else {
    "no_observer_simple"
  }
  simple_dir <- file.path(output_root, "evidence_accumulation_compare", simple_dir_name)
  dir.create(simple_dir, recursive = TRUE, showWarnings = FALSE)
  simple_stem <- file.path(
    simple_dir,
    sprintf(
      "evidence_accumulation_beta_opp_comparison_obsstd_%s_coherence_%s%s",
      value_token(simple_fixed_obsstd),
      values_token(simple_coherence_values),
      paste0(
        "_correctreward_",
        values_token(unique(file_index$correct_reward[is.finite(file_index$correct_reward)])),
        if (pay_kl_on_stop_mode) "_stop_paid" else ""
      )
    )
  )
  message(sprintf(
    "Saving fixed-std/simple evidence plots to %s with obsstd=%s and coherence rows=%s.",
    simple_dir,
    num_label(simple_fixed_obsstd),
    values_label(simple_coherence_values)
  ))

  simple_kl_spec <- list(
    name = "KL paid by timestep",
    slug = "kl_paid_by_timestep",
    summary = kl_timestep_by_coherence_summary,
    x_col = "timestep_before_stopping",
    y_col = "kl_paid",
    xlab = "Steps relative\nto stopping",
    ylab = "KL paid",
    y_lim = positive_log_axis_limit(kl_timestep_by_coherence_summary, "kl_paid"),
    x_lim = relative_timestep_range(kl_timestep_by_coherence_summary$timestep_before_stopping),
    hline = NULL,
    vline = NULL,
    y_scale = "log10"
  )
  simple_delta_action_logit_spec <- list(
    name = "Abs delta action-aligned terminal logit by timestep",
    slug = "abs_delta_action_aligned_action_logit_by_timestep",
    summary = delta_action_logit_by_coherence_summary,
    x_col = "timestep_before_stopping",
    y_col = "abs_delta_action_aligned_action_logit",
    xlab = "Steps relative\nto stopping",
    ylab = "|delta action-aligned\nterminal logit|",
    y_lim = nonnegative_axis_limit(delta_action_logit_by_coherence_summary, "abs_delta_action_aligned_action_logit"),
    x_lim = relative_timestep_range(delta_action_logit_by_coherence_summary$timestep_before_stopping),
    hline = 0,
    vline = NULL
  )
  simple_delta_mu_spec <- list(
    name = "Abs delta z_mu by timestep",
    slug = "abs_delta_z_mu_by_timestep",
    summary = delta_z_mu_by_coherence_summary,
    x_col = "timestep_before_stopping",
    y_col = "abs_delta_z_mu",
    xlab = "Steps relative\nto stopping",
    ylab = "|delta prior-norm\nz_mu|",
    y_lim = positive_log_axis_limit(delta_z_mu_by_coherence_summary, "abs_delta_z_mu"),
    x_lim = relative_timestep_range(delta_z_mu_by_coherence_summary$timestep_before_stopping),
    hline = 0,
    vline = NULL,
    y_scale = "log10"
  )
  simple_z_sigma_spec <- list(
    name = "Average z_sigma by timestep",
    slug = "average_z_sigma_by_timestep",
    summary = z_sigma_timestep_by_coherence_summary,
    x_col = "timestep_before_stopping",
    y_col = "average_z_sigma",
    xlab = "Steps relative\nto stopping",
    ylab = "Average prior-norm\nz_sigma",
    y_lim = positive_log_axis_limit(z_sigma_timestep_by_coherence_summary, "average_z_sigma"),
    x_lim = relative_timestep_range(z_sigma_timestep_by_coherence_summary$timestep_before_stopping),
    hline = NULL,
    vline = NULL,
    y_scale = "log10"
  )
  simple_threshold_spec <- list(
    name = "Decision threshold by stopping duration",
    slug = "decision_threshold_by_stopping_timestep",
    summary = threshold_by_duration_coherence_summary,
    x_col = "stopping_timestep",
    y_col = "choice_aligned_cumulative_evidence",
    xlab = "Total observations\nbefore stopping",
    ylab = "Choice-aligned cumulative\nevidence at stop",
    y_lim = NULL,
    x_lim = stop_duration_xlim,
    hline = 0,
    vline = NULL
  )
  coherence_label <- function(value) sprintf("coh = %s", num_label(value))
  simple_timestep_specs <- list(simple_delta_action_logit_spec, simple_delta_mu_spec, simple_z_sigma_spec, simple_kl_spec)
  if (!observer_only_mode) {
    simple_timestep_specs <- c(simple_timestep_specs, list(simple_threshold_spec))
  }
  for (spec in simple_timestep_specs) {
    save_simple_two_column_plot(
      spec,
      row_values = simple_coherence_values,
      row_col = "coherence",
      row_label_fn = coherence_label,
      output_file = sprintf("%s_%s.png", simple_stem, spec$slug),
      fixed_obsstd = simple_fixed_obsstd
    )
  }

  chronometric_fixed_summary <- chronometric_summary[
    parameter_equal(chronometric_summary$observation_noise_std, simple_fixed_obsstd),
    ,
    drop = FALSE
  ]
  chronometric_fixed_spec <- list(
    name = "Chronometric",
    slug = "chronometric",
    summary = chronometric_fixed_summary,
    x_col = "coherence",
    y_col = "num_observations",
    xlab = "Coherence magnitude",
    ylab = "Number of observations\nbefore choice",
    y_lim = NULL,
    x_lim = coh_xlim,
    hline = NULL,
    vline = NULL
  )
  psychometric_fixed_summary <- psychometric_summary[
    parameter_equal(psychometric_summary$observation_noise_std, simple_fixed_obsstd),
    ,
    drop = FALSE
  ]
  psychometric_fixed_spec <- list(
    name = "Psychometric",
    slug = "psychometric",
    summary = psychometric_fixed_summary,
    x_col = "signed_coherence",
    y_col = "choose_right",
    xlab = "Signed coherence",
    ylab = "Probability of\nchoosing right",
    y_lim = prob_ylim,
    x_lim = signed_xlim,
    hline = 0.5,
    vline = 0
  )
  save_simple_two_column_plot(
    chronometric_fixed_spec,
    row_values = simple_fixed_obsstd,
    row_col = "observation_noise_std",
    row_label_fn = function(value) sprintf("std = %s", num_label(value)),
    output_file = sprintf("%s_%s.png", simple_stem, chronometric_fixed_spec$slug),
    fixed_obsstd = simple_fixed_obsstd
  )
  save_simple_two_column_plot(
    psychometric_fixed_spec,
    row_values = simple_fixed_obsstd,
    row_col = "observation_noise_std",
    row_label_fn = function(value) sprintf("std = %s", num_label(value)),
    output_file = sprintf("%s_%s.png", simple_stem, psychometric_fixed_spec$slug),
    fixed_obsstd = simple_fixed_obsstd
  )
  invisible(NULL)
}

save_response_locked_coherence_overlay_plots()
save_response_locked_target_probability_overlay_plots()
save_chronological_cumulative_evidence_coherence_overlay_plots()
invisible(lapply(analysis_specs, save_analysis_plot))
save_simple_evidence_plots()
