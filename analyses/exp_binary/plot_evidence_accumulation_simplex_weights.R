#!/usr/bin/env Rscript

raw_command_args <- commandArgs(trailingOnly = TRUE)
args <- raw_command_args

script_file_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
script_dir <- if (length(script_file_arg) > 0L) {
  dirname(normalizePath(sub("^--file=", "", script_file_arg[[1]]), mustWork = FALSE))
} else {
  "analyses/exp_binary"
}

usage <- function() {
  cat(
    "Usage:\n",
    "  Rscript analyses/exp_binary/plot_evidence_accumulation_simplex_weights.R [evidence] [options]\n\n",
    "Fits a nonnegative simplex temporal evidence kernel to the final A/B policy\n",
    "output of evidence-accumulation fixed-duration simulations.\n\n",
    "Input conventions intentionally mirror plot_evidence_accumulation_integration_weights.R.\n\n",
    "Options:\n",
    "  --preset-file PATH          Preset CSV path. Default: analyses/exp_binary/evidence_accumulation_plot_presets.csv.\n",
    "  --input-dir DIR             CSV directory. Default: fixed-duration directory derived from preset.\n",
    "  --use-training-simulations  Read regular post-training simulation CSVs instead of fixed-duration wide CSVs.\n",
    "  --output-root DIR           Output root. Default: results.\n",
    "  --output-dir DIR            Exact output base directory. Default: results/evidence_accumulation_simplex_weights.\n",
    "  --vary-memory-lambda-values LIST\n",
    "                              Memory-lambda values for the memory-vary column.\n",
    "                              Aliases: --memory-lambda-values, --memory-lambdas.\n",
    "  --vary-beta-values LIST     Legacy alias for --vary-memory-lambda-values.\n",
    "                              Aliases: --beta-values, --betas.\n",
    "  --vary-opportunity-values LIST\n",
    "                              Opportunity costs for the opportunity-vary column.\n",
    "                              Aliases: --opportunity-values, --opportunities, --opportunity-costs.\n",
    "  --fixed-opp VALUE           Opportunity held fixed for memory-lambda-vary curves.\n",
    "                              Alias: --fixed-opportunity, --fixed-opportunity-cost.\n",
    "  --fixed-memory-lambda VALUE Memory lambda held fixed for opportunity-vary curves.\n",
    "  --fixed-beta VALUE          Legacy alias for --fixed-memory-lambda.\n",
    "  --fixed-coherence VALUE     Coherence magnitude to fit. Alias: --coherence.\n",
    "  --pool-coherence            Pool coherence magnitudes within a fit.\n",
    "  --observation-noise-std LIST\n",
    "                              Observation noise std value(s). Aliases: --obsstd, --sigma,\n",
    "                              --fixed-observation-noise-std.\n",
    "  --simple-fixed-obsstd VALUE Alias for selecting one observation-noise std for simple plots.\n",
    "                              Aliases: --simple-obsstd, --fixed-std-simple, --fixed-obsstd-simple.\n",
    "  --simple-coherence-values LIST\n",
    "                              Coherence magnitudes to include together for simple plots.\n",
    "                              Alias: --simple-coherences.\n",
    "  --loss-scale VALUE          Filter loss scale. Aliases: --lambda, --lambda-value.\n",
    "  --correct-reward VALUE      Filter correct terminal reward scale. Default comes from preset.\n",
    "  --input-type VALUE          Filter trailing input type. Default comes from preset.\n",
    "  --pay-kl-on-stop            Use CSVs with the _stop_paid filename suffix. Default comes from preset.\n",
    "  --no-pay-kl-on-stop         Use legacy CSVs without the _stop_paid filename suffix.\n",
    "  --observer-only             Use observer/end-choice checkpoints only.\n",
    "  --non-observer              Use non-observer/self-timed checkpoints only.\n",
    "  --alpha VALUE               Filter alpha. Default: no filter.\n",
    "  --seeds LIST                Filter seed values. Default: preset seeds/all available.\n",
    "  --checkpoints LIST          Filter checkpoint labels. Default: all available.\n",
    "  --fixed-checkpoint VALUE    Alias for --checkpoints VALUE.\n",
    "  --rnn-units VALUE           Filter RNN units if filename metadata is available.\n",
    "  --latent-dim VALUE          Filter latent dimension if filename metadata is available.\n",
    "  --max-observations VALUE    Number of observations. Alias: --maxobs. Default: 10.\n",
    "  --comparison-mode MODE      both, memory_lambda, opportunity, or checkpoint. Default: both.\n",
    "  --target-type TYPE          auto, logit, probability, or sampled_choice. Default: auto.\n",
    "  --min-trials-per-fit N      Minimum trials per independent simplex fit. Default: 1000.\n",
    "  --num-random-starts N       Number of optimizer starts. Default: 20.\n",
    "  --num-cv-folds N            K-fold CV folds within each run. Default: 5.\n",
    "  --seed N                    Random seed for starts and folds. Default: 1.\n",
    "  --save-predictions          Save trial-level fitted/predicted outputs.\n",
    "  --run-tests                 Run synthetic validation tests and exit.\n",
    "  --help                      Show this message.\n",
    sep = ""
  )
}

if (any(args %in% c("--help", "-h"))) {
  usage()
  quit(save = "no", status = 0L)
}

trim_string <- function(value) trimws(as.character(value))

parse_csv_values <- function(value) {
  if (is.null(value) || !nzchar(trim_string(value))) return(character())
  out <- unlist(strsplit(as.character(value), "[,[:space:]]+"), use.names = FALSE)
  out <- trimws(out)
  out[nzchar(out)]
}

as_num <- function(value) suppressWarnings(as.numeric(as.character(value)))

as_num_token <- function(value) {
  text <- as.character(value)
  out <- suppressWarnings(as.numeric(text))
  bad <- !is.finite(out) & !is.na(text)
  if (any(bad)) {
    text2 <- gsub("p", ".", text[bad], fixed = TRUE)
    out[bad] <- suppressWarnings(as.numeric(text2))
  }
  out
}

num_label <- function(value) {
  value_num <- suppressWarnings(as.numeric(value))
  if (!is.finite(value_num)) return(as.character(value))
  format(signif(value_num, 7), scientific = FALSE, trim = TRUE)
}

value_token <- function(value) {
  token <- gsub("[^A-Za-z0-9]+", "p", num_label(value))
  gsub("^p|p$", "", token)
}

parameter_equal <- function(x, value, tol = 1e-6) {
  abs(as_num(x) - as_num(value)) <= tol
}

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
      if (i == length(args)) stop(sprintf("%s requires a value.", matched))
      value <- args[[i + 1L]]
      keep[[i]] <- FALSE
      keep[[i + 1L]] <- FALSE
      i <- i + 2L
    }
  }
  list(args = args[keep], value = value)
}

extract_boolean_option <- function(args, true_names, default = FALSE) {
  value <- default
  keep <- rep(TRUE, length(args))
  for (i in seq_along(args)) {
    if (args[[i]] %in% true_names) {
      value <- TRUE
      keep[[i]] <- FALSE
    }
  }
  list(args = args[keep], value = value)
}

has_any_option <- function(command_args, option_names) {
  for (arg in command_args) {
    for (option_name in option_names) {
      if (identical(arg, option_name) || startsWith(arg, paste0(option_name, "="))) return(TRUE)
    }
  }
  FALSE
}

parse_bool_value <- function(value, default = FALSE, label = "boolean value") {
  if (is.null(value) || length(value) == 0L || is.na(value) || !nzchar(trim_string(value))) return(default)
  raw <- tolower(trim_string(value))
  if (raw %in% c("1", "true", "t", "yes", "y", "on")) return(TRUE)
  if (raw %in% c("0", "false", "f", "no", "n", "off")) return(FALSE)
  stop(sprintf("Could not parse %s as true/false: %s", label, value))
}

opt <- extract_named_option(args, c("--preset-file"), file.path(script_dir, "evidence_accumulation_plot_presets.csv"))
args <- opt$args
preset_file_arg <- opt$value

opt <- extract_named_option(args, c("--input-dir"), "outputs/jax_simulations_evi_fixed_duration")
args <- opt$args
input_dir <- opt$value

opt <- extract_boolean_option(args, c("--use-training-simulations", "--use-regular-simulations", "--regular-simulations"), FALSE)
args <- opt$args
use_training_simulations <- isTRUE(opt$value)

opt <- extract_named_option(args, c("--output-root", "--results-dir"), "results")
args <- opt$args
output_root <- opt$value

opt <- extract_named_option(args, c("--output-dir"), NULL)
args <- opt$args
output_dir_arg <- opt$value

opt <- extract_named_option(args, c("--vary-memory-lambda-values", "--memory-lambda-values", "--memory-lambdas"), NULL)
args <- opt$args
memory_lambda_values_arg <- opt$value

opt <- extract_named_option(args, c("--vary-beta-values", "--beta-values", "--betas"), NULL)
args <- opt$args
beta_values_arg <- opt$value

opt <- extract_named_option(args, c("--vary-opportunity-values", "--opportunity-values", "--opportunities", "--opportunity-costs"), NULL)
args <- opt$args
opportunity_values_arg <- opt$value

opt <- extract_named_option(args, c("--fixed-opp", "--fixed-opportunity", "--fixed-opportunity-cost"), NULL)
args <- opt$args
fixed_opp_arg <- opt$value

opt <- extract_named_option(args, c("--fixed-memory-lambda", "--fixed-memory-lambda-value"), NULL)
args <- opt$args
fixed_memory_lambda_arg <- opt$value

opt <- extract_named_option(args, c("--fixed-beta"), NULL)
args <- opt$args
fixed_beta_arg <- opt$value

opt <- extract_named_option(args, c("--fixed-coherence", "--coherence"), NULL)
args <- opt$args
fixed_coherence_arg <- opt$value

opt <- extract_boolean_option(args, c("--pool-coherence"), FALSE)
args <- opt$args
pool_coherence <- opt$value

opt <- extract_named_option(args, c("--observation-noise-std", "--obsstd", "--sigma", "--fixed-observation-noise-std"), NULL)
args <- opt$args
obsstd_arg <- opt$value

opt <- extract_named_option(args, c("--simple-fixed-obsstd", "--simple-obsstd", "--fixed-std-simple", "--fixed-obsstd-simple"), NULL)
args <- opt$args
simple_fixed_obsstd_arg <- opt$value

opt <- extract_named_option(args, c("--simple-coherence-values", "--simple-coherences"), NULL)
args <- opt$args
simple_coherence_values_arg <- opt$value

opt <- extract_named_option(args, c("--loss-scale", "--loss-scale-value", "--lambda", "--lambda-value"), NULL)
args <- opt$args
loss_scale_arg <- opt$value

opt <- extract_named_option(args, c("--correct-reward", "--reward-scale", "--terminal-correct-reward"), NULL)
args <- opt$args
correct_reward_arg <- opt$value

opt <- extract_named_option(args, c("--input-type"), NULL)
args <- opt$args
input_type_arg <- opt$value

opt <- extract_boolean_option(args, c("--pay-kl-on-stop", "--stop-paid"), NA)
args <- opt$args
pay_kl_on_stop_arg <- opt$value
opt <- extract_boolean_option(args, c("--no-pay-kl-on-stop", "--no-stop-paid", "--legacy-no-stop-paid"), FALSE)
args <- opt$args
if (isTRUE(opt$value)) pay_kl_on_stop_arg <- FALSE

opt <- extract_boolean_option(args, c("--observer-only", "--choice-at-end-only", "--observer-end-choice"), NA)
args <- opt$args
observer_only_arg <- opt$value
opt <- extract_boolean_option(args, c("--non-observer", "--self-timed", "--policy-duration"), FALSE)
args <- opt$args
if (isTRUE(opt$value)) observer_only_arg <- FALSE

opt <- extract_named_option(args, c("--alpha"), NULL)
args <- opt$args
alpha_arg <- opt$value

opt <- extract_named_option(args, c("--seeds"), NULL)
args <- opt$args
seeds_arg <- opt$value
explicit_seeds_arg <- has_any_option(raw_command_args, c("--seeds"))

opt <- extract_named_option(args, c("--checkpoints", "--fixed-checkpoint"), NULL)
args <- opt$args
checkpoints_arg <- opt$value

opt <- extract_named_option(args, c("--rnn-units", "--rnn-dims", "--rnn-dim"), NULL)
args <- opt$args
rnn_units_arg <- opt$value

opt <- extract_named_option(args, c("--latent-dim", "--latent-dims"), NULL)
args <- opt$args
latent_dim_arg <- opt$value

opt <- extract_named_option(args, c("--max-observations", "--max-observations-before-stop", "--maxobs"), "10")
args <- opt$args
max_observations_arg <- opt$value

opt <- extract_named_option(args, c("--comparison-mode"), "both")
args <- opt$args
comparison_mode <- tolower(trim_string(opt$value))

opt <- extract_named_option(args, c("--target-type"), "auto")
args <- opt$args
target_type_arg <- tolower(trim_string(opt$value))

opt <- extract_named_option(args, c("--min-trials-per-fit"), "1000")
args <- opt$args
min_trials_per_fit <- as.integer(as_num(opt$value))

opt <- extract_named_option(args, c("--num-random-starts", "--random-starts"), "20")
args <- opt$args
num_random_starts <- as.integer(as_num(opt$value))

opt <- extract_named_option(args, c("--num-cv-folds", "--cv-folds"), "5")
args <- opt$args
num_cv_folds <- as.integer(as_num(opt$value))

opt <- extract_named_option(args, c("--seed", "--random-seed"), "1")
args <- opt$args
analysis_seed <- as.integer(as_num(opt$value))

opt <- extract_boolean_option(args, c("--save-predictions"), FALSE)
args <- opt$args
save_predictions <- isTRUE(opt$value)

opt <- extract_boolean_option(args, c("--run-tests"), FALSE)
args <- opt$args
run_tests <- isTRUE(opt$value)

positional <- character()
if (length(args) > 0L && !startsWith(args[[1L]], "-")) {
  positional <- args[[1L]]
  args <- args[-1L]
}
task_name <- if (length(positional) == 0L) "evidence" else positional[[1L]]
if (!identical(task_name, "evidence")) stop(sprintf("Only the evidence task is supported, not: %s", task_name))
if (length(args) > 0L) stop(sprintf("Unexpected argument(s): %s", paste(args, collapse = " ")))

if (!comparison_mode %in% c("both", "memory_lambda", "memory", "beta", "opportunity", "checkpoint")) {
  stop(sprintf("Unsupported --comparison-mode: %s", comparison_mode))
}
if (!target_type_arg %in% c("auto", "logit", "probability", "sampled_choice")) {
  stop(sprintf("Unsupported --target-type: %s", target_type_arg))
}
if (!is.finite(min_trials_per_fit) || min_trials_per_fit < 1L) stop("--min-trials-per-fit must be positive.")
if (!is.finite(num_random_starts) || num_random_starts < 1L) stop("--num-random-starts must be positive.")
if (!is.finite(num_cv_folds) || num_cv_folds < 0L) stop("--num-cv-folds must be nonnegative.")

simple_fixed_obsstd <- if (!is.null(simple_fixed_obsstd_arg) && nzchar(trim_string(simple_fixed_obsstd_arg))) {
  vals <- as_num(parse_csv_values(simple_fixed_obsstd_arg))
  vals <- vals[is.finite(vals)]
  if (length(vals) != 1L) stop("--simple-fixed-obsstd expects exactly one numeric value.")
  vals[[1L]]
} else {
  NA_real_
}
simple_coherence_values <- if (!is.null(simple_coherence_values_arg) && nzchar(trim_string(simple_coherence_values_arg))) {
  vals <- as_num(parse_csv_values(simple_coherence_values_arg))
  vals[is.finite(vals)]
} else {
  numeric()
}
simple_plot_mode <- is.finite(simple_fixed_obsstd) || length(simple_coherence_values) > 0L

load_preset_rows <- function(preset_file, task) {
  if (!file.exists(preset_file)) stop(sprintf("Preset file not found: %s", preset_file))
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
  memory_row <- rows[vary_labels %in% c("memory_lambda", "memory", "beta"), , drop = FALSE]
  opp_row <- rows[vary_labels == "opportunity", , drop = FALSE]
  if (nrow(memory_row) == 0L || nrow(opp_row) == 0L) {
    stop(sprintf("Need both memory_lambda and opportunity rows for task=%s in %s.", task, preset_file))
  }
  list(memory = memory_row[1L, , drop = FALSE], opportunity = opp_row[1L, , drop = FALSE])
}

preset_value <- function(row, column, default = NULL) {
  if (!column %in% names(row)) return(default)
  value <- row[[column]][[1L]]
  if (is.na(value) || !nzchar(trim_string(value))) return(default)
  trim_string(value)
}

join_csv_unique <- function(...) {
  values <- unlist(lapply(list(...), parse_csv_values), use.names = FALSE)
  values <- unique(values[nzchar(values)])
  paste(values, collapse = ",")
}

resolve_fixed_duration_input_dir <- function(row) {
  direct <- preset_value(row, "fixed_duration_input_dir",
    preset_value(row, "integration_input_dir",
      preset_value(row, "duration_input_dir", NULL)
    )
  )
  if (!is.null(direct) && nzchar(trim_string(direct))) return(direct)
  base_dir <- preset_value(row, "input_dir", "outputs/jax_simulations_evi")
  candidate <- paste0(base_dir, "_fixed_duration")
  if (dir.exists(candidate)) return(candidate)
  if (grepl("_evi$", base_dir)) {
    candidate2 <- sub("_evi$", "_evi_fixed_duration", base_dir)
    if (dir.exists(candidate2)) return(candidate2)
  }
  "outputs/jax_simulations_evi_fixed_duration"
}

preset_rows <- load_preset_rows(preset_file_arg, task_name)
preset_memory_row <- preset_rows$memory
preset_opp_row <- preset_rows$opportunity
memory_family_seeds_arg <- preset_value(preset_memory_row, "seed_arg", NULL)
opportunity_family_seeds_arg <- preset_value(preset_opp_row, "seed_arg", NULL)

if (!has_any_option(raw_command_args, c("--input-dir"))) {
  input_dir <- if (use_training_simulations) {
    preset_value(preset_memory_row, "input_dir", "outputs/jax_simulations_evi")
  } else {
    resolve_fixed_duration_input_dir(preset_memory_row)
  }
}
if (!has_any_option(raw_command_args, c("--output-root", "--results-dir"))) {
  output_root <- preset_value(preset_memory_row, "results_dir", output_root)
}
if (!has_any_option(raw_command_args, c("--input-type"))) {
  input_type_arg <- preset_value(preset_memory_row, "input_type", input_type_arg)
}
if (!has_any_option(raw_command_args, c("--vary-memory-lambda-values", "--memory-lambda-values", "--memory-lambdas", "--vary-beta-values", "--beta-values", "--betas"))) {
  memory_lambda_values_arg <- preset_value(preset_memory_row, "memory_lambda_arg", preset_value(preset_memory_row, "beta_arg", memory_lambda_values_arg))
}
if (!has_any_option(raw_command_args, c("--vary-opportunity-values", "--opportunity-values", "--opportunities", "--opportunity-costs"))) {
  opportunity_values_arg <- preset_value(preset_opp_row, "opportunity_arg", opportunity_values_arg)
}
if (!has_any_option(raw_command_args, c("--fixed-opp", "--fixed-opportunity", "--fixed-opportunity-cost"))) {
  fixed_opp_arg <- preset_value(preset_memory_row, "opportunity_arg", fixed_opp_arg)
}
if (!has_any_option(raw_command_args, c("--fixed-memory-lambda", "--fixed-memory-lambda-value", "--fixed-beta"))) {
  fixed_memory_lambda_arg <- preset_value(preset_opp_row, "memory_lambda_arg", preset_value(preset_opp_row, "beta_arg", fixed_memory_lambda_arg))
}
if (is.finite(simple_fixed_obsstd) && !has_any_option(raw_command_args, c("--observation-noise-std", "--obsstd", "--sigma", "--fixed-observation-noise-std"))) {
  obsstd_arg <- as.character(simple_fixed_obsstd)
} else if (!has_any_option(raw_command_args, c("--observation-noise-std", "--obsstd", "--sigma", "--fixed-observation-noise-std"))) {
  obsstd_arg <- preset_value(preset_memory_row, "observation_noise_std_arg", obsstd_arg)
}
if (!has_any_option(raw_command_args, c("--loss-scale", "--loss-scale-value", "--lambda", "--lambda-value"))) {
  loss_scale_arg <- preset_value(preset_memory_row, "loss_scale_arg", preset_value(preset_memory_row, "lambda_arg", loss_scale_arg))
}
if (!has_any_option(raw_command_args, c("--alpha"))) {
  alpha_arg <- preset_value(preset_memory_row, "alpha_arg", alpha_arg)
}
if (!explicit_seeds_arg) {
  seeds_arg <- if (comparison_mode %in% c("memory_lambda", "memory", "beta")) {
    memory_family_seeds_arg
  } else if (comparison_mode %in% c("opportunity")) {
    opportunity_family_seeds_arg
  } else {
    join_csv_unique(memory_family_seeds_arg, opportunity_family_seeds_arg)
  }
  if (!nzchar(seeds_arg)) seeds_arg <- NULL
}
if (!has_any_option(raw_command_args, c("--rnn-units", "--rnn-dims", "--rnn-dim"))) {
  rnn_units_arg <- preset_value(preset_memory_row, "rnn_units_arg", rnn_units_arg)
}
if (!has_any_option(raw_command_args, c("--latent-dim", "--latent-dims"))) {
  latent_dim_arg <- preset_value(preset_memory_row, "latent_dim_arg", latent_dim_arg)
}
if (!has_any_option(raw_command_args, c("--max-observations", "--max-observations-before-stop", "--maxobs"))) {
  max_observations_arg <- preset_value(preset_memory_row, "max_observations_arg", max_observations_arg)
}
if (!has_any_option(raw_command_args, c("--correct-reward", "--reward-scale", "--terminal-correct-reward"))) {
  correct_reward_arg <- preset_value(preset_memory_row, "correct_reward_arg", correct_reward_arg)
}
if (!has_any_option(raw_command_args, c("--pay-kl-on-stop", "--stop-paid", "--no-pay-kl-on-stop", "--no-stop-paid", "--legacy-no-stop-paid"))) {
  pay_kl_on_stop_arg <- parse_bool_value(
    preset_value(preset_memory_row, "pay_kl_on_stop_arg", NA),
    default = pay_kl_on_stop_arg,
    label = "pay_kl_on_stop_arg"
  )
}
if (!has_any_option(raw_command_args, c("--observer-only", "--choice-at-end-only", "--observer-end-choice", "--non-observer", "--self-timed", "--policy-duration"))) {
  observer_only_arg <- parse_bool_value(
    preset_value(preset_memory_row, "observer_only_arg", NA),
    default = observer_only_arg,
    label = "observer_only_arg"
  )
}
if (!pool_coherence && length(simple_coherence_values) == 0L && !has_any_option(raw_command_args, c("--fixed-coherence", "--coherence", "--pool-coherence"))) {
  preset_coherence_values <- parse_csv_values(preset_value(preset_memory_row, "coherence_arg", ""))
  if (length(preset_coherence_values) == 1L) fixed_coherence_arg <- preset_coherence_values[[1L]]
}

max_observations <- as.integer(as_num(max_observations_arg))
if (!is.finite(max_observations) || max_observations < 1L) stop("--max-observations must be positive.")
message(sprintf("Using evidence accumulation preset: task=%s from %s", task_name, preset_file_arg))

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

sem_or_na <- function(x) {
  x <- as_num(x)
  x <- x[is.finite(x)]
  if (length(x) <= 1L) return(NA_real_)
  stats::sd(x) / sqrt(length(x))
}

rbind_fill <- function(rows) {
  rows <- rows[!vapply(rows, is.null, logical(1))]
  if (length(rows) == 0L) return(data.frame())
  all_names <- unique(unlist(lapply(rows, names), use.names = FALSE))
  aligned <- lapply(rows, function(dat) {
    missing <- setdiff(all_names, names(dat))
    for (nm in missing) dat[[nm]] <- NA
    dat[, all_names, drop = FALSE]
  })
  do.call(rbind, aligned)
}

mode_numeric <- function(values) {
  values <- as_num(values)
  values <- values[is.finite(values)]
  if (length(values) == 0L) return(NA_real_)
  tab <- sort(table(signif(values, 12)), decreasing = TRUE)
  as.numeric(names(tab)[[1L]])
}

safe_range <- function(values, pad_fraction = 0.05, fallback = c(0, 1)) {
  values <- as_num(values)
  values <- values[is.finite(values)]
  if (length(values) == 0L) return(fallback)
  lim <- range(values)
  if (abs(diff(lim)) < 1e-12) lim <- lim + c(-0.5, 0.5)
  pad <- diff(lim) * pad_fraction
  lim + c(-pad, pad)
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

parse_model_metadata <- function(path) {
  base <- basename(path)
  field <- function(prefix) {
    pattern <- paste0("(?:^|_)", prefix, "_([^_]+)")
    match <- regexec(pattern, base, perl = TRUE)
    hit <- regmatches(base, match)[[1]]
    if (length(hit) < 2L) return(NA_character_)
    hit[[2L]]
  }
  checkpoint_match <- regexec("_checkpoint_([^_]+)", base, perl = TRUE)
  checkpoint_hit <- regmatches(base, checkpoint_match)[[1]]
  checkpoint <- if (length(checkpoint_hit) >= 2L) checkpoint_hit[[2L]] else "final"
  loss_scale <- as_num_token(field("loss_scale"))
  legacy_lambda <- as_num_token(field("lambda"))
  if (!is.finite(loss_scale)) loss_scale <- legacy_lambda
  beta <- as_num_token(field("beta"))
  memory_lambda <- as_num_token(field("memorylambda"))
  if (!is.finite(memory_lambda) && is.finite(beta)) memory_lambda <- 1 / beta
  opportunity_cost <- as_num_token(field("opportunity"))
  if (!is.finite(opportunity_cost)) opportunity_cost <- as_num_token(field("opp"))
  correct_reward <- as_num_token(field("correctreward"))
  input_type_match <- regexec("_([^_]+)_wide\\.csv$", base, perl = TRUE)
  input_type_hit <- regmatches(base, input_type_match)[[1]]
  if (length(input_type_hit) < 2L) {
    input_type_match <- regexec("_([^_]+)\\.csv$", base, perl = TRUE)
    input_type_hit <- regmatches(base, input_type_match)[[1]]
  }
  input_type <- if (length(input_type_hit) >= 2L) input_type_hit[[2L]] else NA_character_
  data.frame(
    source_file = path,
    source_format = if (grepl("_wide\\.csv$", base, perl = TRUE)) "duration_wide" else "training_sim",
    checkpoint = checkpoint,
    loss_scale = loss_scale,
    lambda = loss_scale,
    memory_lambda = memory_lambda,
    choice_at_end_only = grepl("_observer_endchoice(?:_|$)", base, perl = TRUE),
    duration_mode = if (grepl("_policy_duration_", base, perl = TRUE)) {
      "policy"
    } else if (grepl("_fixed_duration_", base, perl = TRUE)) {
      "fixed"
    } else {
      "training_sim"
    },
    alpha = as_num_token(field("alpha")),
    beta = beta,
    opportunity_cost = opportunity_cost,
    correct_reward = correct_reward,
    pay_kl_on_stop = grepl("_stop_paid(?:_|$)", base, perl = TRUE),
    seed = as_num_token(field("seed")),
    observation_noise_std = as_num_token(field("obsstd")),
    max_observations_before_stop = as_num_token(field("maxobs")),
    rnn_units = as_num_token(field("rnn")),
    latent_dim = as_num_token(field("latent")),
    input_type = input_type,
    stringsAsFactors = FALSE
  )
}

numeric_arg_values <- function(value) {
  values <- as_num_token(parse_csv_values(value))
  values[is.finite(values)]
}

seed_values_for_family <- function(family) {
  if (explicit_seeds_arg) return(numeric_arg_values(seeds_arg))
  if (identical(family, "beta")) return(numeric_arg_values(memory_family_seeds_arg))
  if (identical(family, "opportunity")) return(numeric_arg_values(opportunity_family_seeds_arg))
  numeric_arg_values(seeds_arg)
}

data_matches_seed_values <- function(values, requested) {
  if (length(requested) == 0L) return(rep(TRUE, length(values)))
  keep <- rep(FALSE, length(values))
  for (v in requested) keep <- keep | parameter_equal(values, v)
  keep
}

metadata_matches_values <- function(values, requested, unknown_ok = TRUE) {
  if (length(requested) == 0L) return(rep(TRUE, length(values)))
  x <- as_num(values)
  known <- is.finite(x)
  keep <- rep(FALSE, length(x))
  for (v in requested) keep <- keep | (known & parameter_equal(x, v))
  if (unknown_ok) keep <- keep | !known
  keep
}

metadata_matches_strings <- function(values, requested, unknown_ok = TRUE) {
  if (length(requested) == 0L) return(rep(TRUE, length(values)))
  text <- as.character(values)
  known <- !is.na(text) & nzchar(text)
  keep <- known & text %in% requested
  if (unknown_ok) keep <- keep | !known
  keep
}

stable_softmax <- function(theta) {
  theta <- as_num(theta)
  theta <- theta - mean(theta)
  z <- theta - max(theta)
  e <- exp(z)
  e / sum(e)
}

softplus <- function(x) log1p(exp(-abs(x))) + pmax(x, 0)

inverse_softplus <- function(y) {
  y <- pmax(as_num(y), 1e-8)
  ifelse(y > 30, y, log(expm1(y)))
}

stable_sigmoid <- function(x) {
  x <- as_num(x)
  out <- numeric(length(x))
  pos <- x >= 0
  out[pos] <- 1 / (1 + exp(-x[pos]))
  ex <- exp(x[!pos])
  out[!pos] <- ex / (1 + ex)
  out
}

predict_simplex_logit <- function(par, obs_mat) {
  n_obs <- ncol(obs_mat)
  theta <- par[seq_len(n_obs)]
  gain <- softplus(par[[n_obs + 1L]])
  bias <- par[[n_obs + 2L]]
  weights <- stable_softmax(theta)
  as.vector(bias + gain * (obs_mat %*% weights))
}

extract_simplex_params <- function(par, n_obs) {
  weights <- stable_softmax(par[seq_len(n_obs)])
  gain <- softplus(par[[n_obs + 1L]])
  bias <- par[[n_obs + 2L]]
  list(weights = weights, gain = gain, bias = bias, effective = gain * weights)
}

simplex_loss <- function(par, obs_mat, target, target_type) {
  pred_logit <- predict_simplex_logit(par, obs_mat)
  if (!all(is.finite(pred_logit))) return(Inf)
  if (identical(target_type, "logit")) {
    return(mean((as_num(target) - pred_logit)^2))
  }
  pred_p <- pmin(pmax(stable_sigmoid(pred_logit), 1e-8), 1 - 1e-8)
  y <- pmin(pmax(as_num(target), 1e-8), 1 - 1e-8)
  -mean(y * log(pred_p) + (1 - y) * log(1 - pred_p))
}

linear_initialization <- function(obs_mat, target, target_type) {
  y <- if (identical(target_type, "logit")) {
    as_num(target)
  } else {
    stats::qlogis(pmin(pmax(as_num(target), 1e-6), 1 - 1e-6))
  }
  dat <- as.data.frame(obs_mat)
  names(dat) <- paste0("observation_", seq_len(ncol(obs_mat)))
  dat$target <- y
  fit <- tryCatch(stats::lm(target ~ ., data = dat), error = function(e) NULL)
  if (is.null(fit)) return(NULL)
  coef <- stats::coef(fit)[paste0("observation_", seq_len(ncol(obs_mat)))]
  coef[!is.finite(coef)] <- 0
  pos <- pmax(as_num(coef), 0)
  if (sum(pos) <= 0) pos <- rep(1, length(coef))
  w <- pos / sum(pos)
  gain <- max(sum(pos), 1e-6)
  b <- as_num(stats::coef(fit)[["(Intercept)"]])
  if (!is.finite(b)) b <- mean(y, na.rm = TRUE)
  c(log(pmax(w, 1e-8)), inverse_softplus(gain), b)
}

fit_simplex_model <- function(obs_mat, target, target_type, num_starts, seed) {
  n_obs <- ncol(obs_mat)
  set.seed(seed)
  target_for_bias <- if (identical(target_type, "logit")) {
    as_num(target)
  } else {
    stats::qlogis(pmin(pmax(as_num(target), 1e-6), 1 - 1e-6))
  }
  total_evidence <- rowSums(obs_mat)
  slope <- suppressWarnings(stats::coef(stats::lm(target_for_bias ~ total_evidence))[["total_evidence"]])
  if (!is.finite(slope) || slope < 0) slope <- 1e-3
  bias0 <- mean(target_for_bias, na.rm = TRUE)
  starts <- list(c(rep(0, n_obs), inverse_softplus(slope * n_obs), bias0))
  lin <- linear_initialization(obs_mat, target, target_type)
  if (!is.null(lin)) starts[[length(starts) + 1L]] <- lin
  while (length(starts) < num_starts) {
    starts[[length(starts) + 1L]] <- c(rnorm(n_obs, 0, 0.1), inverse_softplus(abs(slope * n_obs) + runif(1, 1e-4, 1)), bias0 + rnorm(1, 0, 0.1))
  }
  fits <- lapply(seq_len(num_starts), function(i) {
    init <- starts[[i]]
    res <- tryCatch(
      stats::optim(
        init,
        simplex_loss,
        obs_mat = obs_mat,
        target = target,
        target_type = target_type,
        method = "BFGS",
        control = list(maxit = 1000, reltol = 1e-10)
      ),
      error = function(e) e
    )
    if (inherits(res, "error")) {
      return(list(value = Inf, par = rep(NA_real_, n_obs + 2L), convergence = NA_integer_, message = conditionMessage(res), counts = NA_integer_))
    }
    list(value = res$value, par = res$par, convergence = res$convergence, message = if (is.null(res$message)) "" else res$message, counts = unname(res$counts[["function"]]))
  })
  finite <- vapply(fits, function(x) is.finite(x$value), logical(1))
  if (!any(finite)) return(NULL)
  best_idx <- which.min(vapply(fits, function(x) x$value, numeric(1)))
  best <- fits[[best_idx]]
  start_values <- vapply(fits, function(x) x$value, numeric(1))
  best$start_objectives <- start_values
  best$n_finite_starts <- sum(finite)
  best$start_objective_sd <- stats::sd(start_values[is.finite(start_values)])
  best
}

evaluate_simplex <- function(par, obs_mat, target, target_type) {
  pred_logit <- predict_simplex_logit(par, obs_mat)
  if (identical(target_type, "logit")) {
    y <- as_num(target)
    mse <- mean((y - pred_logit)^2)
    rmse <- sqrt(mse)
    mae <- mean(abs(y - pred_logit))
    ss_tot <- sum((y - mean(y))^2)
    r2 <- if (ss_tot > 0) 1 - sum((y - pred_logit)^2) / ss_tot else NA_real_
    pearson <- suppressWarnings(stats::cor(y, pred_logit, method = "pearson"))
    spearman <- suppressWarnings(stats::cor(y, pred_logit, method = "spearman"))
    return(list(loss = mse, mse = mse, rmse = rmse, mae = mae, r2 = r2, pearson = pearson, spearman = spearman))
  }
  y <- pmin(pmax(as_num(target), 1e-8), 1 - 1e-8)
  p <- pmin(pmax(stable_sigmoid(pred_logit), 1e-8), 1 - 1e-8)
  ce <- -mean(y * log(p) + (1 - y) * log(1 - p))
  kl <- mean(y * log(y / p) + (1 - y) * log((1 - y) / (1 - p)))
  brier <- mean((y - p)^2)
  corr <- suppressWarnings(stats::cor(y, p))
  list(loss = ce, cross_entropy = ce, bernoulli_kl = kl, brier = brier, prob_correlation = corr, r2 = NA_real_, rmse = sqrt(brier), mae = mean(abs(y - p)))
}

fit_unconstrained_model <- function(obs_mat, target, target_type) {
  y <- if (identical(target_type, "logit")) {
    as_num(target)
  } else {
    stats::qlogis(pmin(pmax(as_num(target), 1e-6), 1 - 1e-6))
  }
  dat <- as.data.frame(obs_mat)
  names(dat) <- paste0("observation_", seq_len(ncol(obs_mat)))
  dat$target <- y
  fit <- tryCatch(stats::lm(target ~ ., data = dat), error = function(e) NULL)
  if (is.null(fit)) return(NULL)
  pred <- as_num(stats::predict(fit, newdata = dat))
  coef <- stats::coef(fit)[paste0("observation_", seq_len(ncol(obs_mat)))]
  coef[!is.finite(coef)] <- NA_real_
  ss_tot <- sum((y - mean(y))^2)
  list(
    coefficients = as_num(coef),
    r2 = if (ss_tot > 0) 1 - sum((y - pred)^2) / ss_tot else NA_real_,
    rmse = sqrt(mean((y - pred)^2)),
    negative_count = sum(coef < 0, na.rm = TRUE)
  )
}

cross_validate_simplex_model <- function(obs_mat, target, target_type, folds, num_starts, seed) {
  if (folds <= 1L) return(list(r2 = NA_real_, rmse = NA_real_, loss = NA_real_, fold_rows = data.frame()))
  n <- nrow(obs_mat)
  if (n < folds) return(list(r2 = NA_real_, rmse = NA_real_, loss = NA_real_, fold_rows = data.frame()))
  set.seed(seed)
  fold_id <- sample(rep(seq_len(folds), length.out = n))
  fold_rows <- list()
  pred_all <- rep(NA_real_, n)
  for (fold in seq_len(folds)) {
    train <- fold_id != fold
    test <- fold_id == fold
    fit <- fit_simplex_model(obs_mat[train, , drop = FALSE], target[train], target_type, num_starts, seed + fold * 1009L)
    if (is.null(fit)) next
    pred <- predict_simplex_logit(fit$par, obs_mat[test, , drop = FALSE])
    pred_all[test] <- pred
    ev <- evaluate_simplex(fit$par, obs_mat[test, , drop = FALSE], target[test], target_type)
    pars <- extract_simplex_params(fit$par, ncol(obs_mat))
    fold_rows[[length(fold_rows) + 1L]] <- data.frame(
      fold = fold,
      n_train = sum(train),
      n_test = sum(test),
      cv_loss = ev$loss,
      cv_R2 = if (!is.null(ev$r2)) ev$r2 else NA_real_,
      cv_RMSE = if (!is.null(ev$rmse)) ev$rmse else NA_real_,
      gain = pars$gain,
      bias = pars$bias
    )
  }
  ok <- is.finite(pred_all)
  if (identical(target_type, "logit")) {
    y <- as_num(target)
    ss_tot <- sum((y[ok] - mean(y[ok]))^2)
    r2 <- if (sum(ok) > 1L && ss_tot > 0) 1 - sum((y[ok] - pred_all[ok])^2) / ss_tot else NA_real_
    rmse <- if (sum(ok) > 0L) sqrt(mean((y[ok] - pred_all[ok])^2)) else NA_real_
    loss <- if (sum(ok) > 0L) mean((y[ok] - pred_all[ok])^2) else NA_real_
  } else {
    y <- pmin(pmax(as_num(target), 1e-8), 1 - 1e-8)
    p <- pmin(pmax(stable_sigmoid(pred_all), 1e-8), 1 - 1e-8)
    loss <- if (sum(ok) > 0L) -mean(y[ok] * log(p[ok]) + (1 - y[ok]) * log(1 - p[ok])) else NA_real_
    rmse <- if (sum(ok) > 0L) sqrt(mean((y[ok] - p[ok])^2)) else NA_real_
    r2 <- NA_real_
  }
  list(r2 = r2, rmse = rmse, loss = loss, fold_rows = if (length(fold_rows) > 0L) do.call(rbind, fold_rows) else data.frame())
}

run_synthetic_test <- function() {
  set.seed(42)
  n <- 4000L
  n_obs <- 10L
  obs <- matrix(rnorm(n * n_obs), nrow = n)
  true_weights <- stable_softmax(c(-2, -1, 0, 1, 2, 2, 1, 0, -1, -2))
  true_gain <- 3
  true_bias <- -0.2
  q <- as.vector(true_bias + true_gain * (obs %*% true_weights))
  fit <- fit_simplex_model(obs, q, "logit", 8L, 10L)
  if (is.null(fit)) stop("Synthetic simplex fit failed.")
  pars <- extract_simplex_params(fit$par, n_obs)
  if (max(abs(sum(pars$weights) - 1)) > 1e-8) stop("Synthetic weights do not sum to one.")
  if (max(abs(pars$weights - true_weights)) > 0.05) stop("Synthetic weights were not recovered.")
  if (abs(pars$gain - true_gain) > 0.25) stop("Synthetic gain was not recovered.")
  if (any(pars$weights < -1e-12) || pars$gain < 0) stop("Synthetic constraints failed.")
  p <- stable_sigmoid(q)
  fit_p <- fit_simplex_model(obs, p, "probability", 8L, 11L)
  if (is.null(fit_p)) stop("Synthetic probability fit failed.")
  pars_p <- extract_simplex_params(fit_p$par, n_obs)
  if (stats::cor(pars$weights, pars_p$weights) < 0.95) stop("Logit/probability synthetic weights disagree.")
  rev_q <- as.vector(-obs %*% true_weights)
  uncon <- fit_unconstrained_model(obs, rev_q, "logit")
  if (is.null(uncon) || uncon$negative_count == 0L) stop("Negative-coefficient diagnostic failed.")
  message("Synthetic simplex validation passed.")
}

if (run_tests) {
  run_synthetic_test()
  quit(save = "no", status = 0L)
}

if (!dir.exists(input_dir)) stop(sprintf("Input directory does not exist: %s", input_dir))

all_files <- if (use_training_simulations) {
  list.files(input_dir, pattern = "_evidence\\.csv$", full.names = TRUE)
} else {
  list.files(input_dir, pattern = "_(fixed|policy)_duration_[0-9]+_checkpoint_.*_wide\\.csv$", full.names = TRUE)
}
if (length(all_files) == 0L) {
  stop(sprintf(
    "No %s CSVs found in %s.",
    if (use_training_simulations) "regular evidence simulation" else "duration-controlled wide",
    input_dir
  ))
}

file_info <- file.info(all_files)
manifest <- do.call(rbind, lapply(all_files, parse_model_metadata))
manifest$file_size <- as_num(file_info$size)
manifest$file_mtime <- as.numeric(file_info$mtime)
prefilter_keep <- is.finite(manifest$file_size) & manifest$file_size > 0
prefilter_keep <- prefilter_keep & metadata_matches_values(manifest$loss_scale, numeric_arg_values(loss_scale_arg))
prefilter_keep <- prefilter_keep & metadata_matches_values(manifest$alpha, numeric_arg_values(alpha_arg))
prefilter_keep <- prefilter_keep & metadata_matches_values(manifest$seed, numeric_arg_values(seeds_arg))
prefilter_keep <- prefilter_keep & metadata_matches_values(manifest$rnn_units, numeric_arg_values(rnn_units_arg))
prefilter_keep <- prefilter_keep & metadata_matches_values(manifest$latent_dim, numeric_arg_values(latent_dim_arg))
prefilter_keep <- prefilter_keep & metadata_matches_values(manifest$max_observations_before_stop, max_observations)
prefilter_keep <- prefilter_keep & metadata_matches_values(manifest$observation_noise_std, numeric_arg_values(obsstd_arg))
prefilter_keep <- prefilter_keep & metadata_matches_values(manifest$correct_reward, numeric_arg_values(correct_reward_arg))
prefilter_keep <- prefilter_keep & metadata_matches_strings(manifest$input_type, parse_csv_values(input_type_arg))
if (!is.na(pay_kl_on_stop_arg)) {
  prefilter_keep <- prefilter_keep & (is.na(manifest$pay_kl_on_stop) | as.logical(manifest$pay_kl_on_stop) == isTRUE(pay_kl_on_stop_arg))
}
if (!is.na(observer_only_arg)) {
  prefilter_keep <- prefilter_keep & (is.na(manifest$choice_at_end_only) | as.logical(manifest$choice_at_end_only) == isTRUE(observer_only_arg))
}
if (!is.null(checkpoints_arg) && nzchar(trim_string(checkpoints_arg))) {
  prefilter_keep <- prefilter_keep & metadata_matches_strings(manifest$checkpoint, parse_csv_values(checkpoints_arg))
}

prefilter_memory_values <- numeric_arg_values(memory_lambda_values_arg)
if (length(prefilter_memory_values) == 0L) prefilter_memory_values <- numeric_arg_values(beta_values_arg)
prefilter_opp_values <- numeric_arg_values(opportunity_values_arg)
prefilter_fixed_opp <- numeric_arg_values(fixed_opp_arg)
prefilter_fixed_memory <- numeric_arg_values(fixed_memory_lambda_arg)
if (length(prefilter_fixed_memory) == 0L) prefilter_fixed_memory <- numeric_arg_values(fixed_beta_arg)
family_filters <- list()
if (length(prefilter_memory_values) > 0L && length(prefilter_fixed_opp) > 0L) {
  filt <-
    metadata_matches_values(manifest$memory_lambda, prefilter_memory_values) &
    metadata_matches_values(manifest$opportunity_cost, prefilter_fixed_opp)
  mem_seed_values <- seed_values_for_family("beta")
  if (length(mem_seed_values) > 0L) {
    filt <- filt & metadata_matches_values(manifest$seed, mem_seed_values)
  }
  family_filters[[length(family_filters) + 1L]] <- filt
} else if (length(prefilter_memory_values) > 0L) {
  filt <- metadata_matches_values(manifest$memory_lambda, prefilter_memory_values)
  mem_seed_values <- seed_values_for_family("beta")
  if (length(mem_seed_values) > 0L) {
    filt <- filt & metadata_matches_values(manifest$seed, mem_seed_values)
  }
  family_filters[[length(family_filters) + 1L]] <- filt
}
if (length(prefilter_opp_values) > 0L && length(prefilter_fixed_memory) > 0L) {
  filt <-
    metadata_matches_values(manifest$opportunity_cost, prefilter_opp_values) &
    metadata_matches_values(manifest$memory_lambda, prefilter_fixed_memory)
  opp_seed_values <- seed_values_for_family("opportunity")
  if (length(opp_seed_values) > 0L) {
    filt <- filt & metadata_matches_values(manifest$seed, opp_seed_values)
  }
  family_filters[[length(family_filters) + 1L]] <- filt
} else if (length(prefilter_opp_values) > 0L) {
  filt <- metadata_matches_values(manifest$opportunity_cost, prefilter_opp_values)
  opp_seed_values <- seed_values_for_family("opportunity")
  if (length(opp_seed_values) > 0L) {
    filt <- filt & metadata_matches_values(manifest$seed, opp_seed_values)
  }
  family_filters[[length(family_filters) + 1L]] <- filt
}
if (length(family_filters) > 0L) prefilter_keep <- prefilter_keep & Reduce(`|`, family_filters)

selected_manifest <- manifest[prefilter_keep, , drop = FALSE]
message(sprintf(
  "Filename/metadata prefilter kept %d/%d %s CSV file(s) before reading rows.",
  nrow(selected_manifest),
  nrow(manifest),
  if (use_training_simulations) "regular evidence" else "wide"
))
if (nrow(selected_manifest) == 0L) stop("No evidence CSVs remain after filename/metadata filters.")

obs_cols <- paste0("observation_", seq_len(max_observations))
evidence_sample_cols <- paste0("evidence_sample_t", seq_len(max_observations))
target_step <- max_observations
target_cols <- c(
  paste0("raw_logit_choose_a_t", target_step),
  paste0("raw_logit_choose_b_t", target_step),
  paste0("choice_logit_t", target_step),
  paste0("p_choose_b_given_terminal_t", target_step),
  paste0("policy_choose_a_t", target_step),
  paste0("policy_choose_b_t", target_step),
  paste0("valid_t", target_step)
)
required_cols <- c(
  "trial_id", "run_id", "seed", "checkpoint", "training_step", "loss_scale", "lambda", "memory_lambda", "alpha", "beta",
  "opportunity_cost", "coherence", "signed_coherence", "observation_noise_std", "correct_choice", "correct_action",
  "terminal_action", "choose_right", "correct", "choose_correct", "num_observations", "graph", "correct_reward",
  "pay_kl_on_stop", "choice_at_end_only", "duration_mode", "input_type", obs_cols, evidence_sample_cols, target_cols
)

loaded <- list()
for (path in selected_manifest$source_file) {
  cols <- read_csv_names(path)
  select <- intersect(required_cols, cols)
  has_observation_cols <- all(obs_cols %in% cols)
  has_evidence_sample_cols <- all(evidence_sample_cols %in% cols)
  if (!has_observation_cols && !has_evidence_sample_cols) {
    warning(sprintf("Skipping %s because it lacks observation/evidence_sample columns.", basename(path)))
    next
  }
  dat <- read_csv_fast(path, select = select)
  meta <- parse_model_metadata(path)
  if (!all(obs_cols %in% names(dat)) && all(evidence_sample_cols %in% names(dat))) {
    for (t in seq_len(max_observations)) dat[[paste0("observation_", t)]] <- dat[[paste0("evidence_sample_t", t)]]
  }
  if (!"trial_id" %in% names(dat)) dat$trial_id <- if ("graph" %in% names(dat)) dat$graph else seq_len(nrow(dat))
  if (!"run_id" %in% names(dat)) dat$run_id <- sub("\\.csv$", "", basename(path))
  if (!"checkpoint" %in% names(dat) || all(is.na(dat$checkpoint) | !nzchar(as.character(dat$checkpoint)))) dat$checkpoint <- meta$checkpoint[[1L]]
  if (!"training_step" %in% names(dat)) dat$training_step <- NA_real_
  if (!"correct" %in% names(dat) && "choose_correct" %in% names(dat)) dat$correct <- dat$choose_correct
  for (nm in names(meta)) {
    if (identical(nm, "source_file")) next
    if (!nm %in% names(dat) || all(!is.finite(as_num(dat[[nm]])))) dat[[nm]] <- meta[[nm]][[1L]]
  }
  if (!"loss_scale" %in% names(dat) || all(!is.finite(as_num(dat$loss_scale)))) dat$loss_scale <- if ("lambda" %in% names(dat)) as_num(dat$lambda) else meta$loss_scale[[1L]]
  if (!"lambda" %in% names(dat) || all(!is.finite(as_num(dat$lambda)))) dat$lambda <- as_num(dat$loss_scale)
  if (!"memory_lambda" %in% names(dat) || all(!is.finite(as_num(dat$memory_lambda)))) dat$memory_lambda <- meta$memory_lambda[[1L]]
  if (!"choice_at_end_only" %in% names(dat)) dat$choice_at_end_only <- meta$choice_at_end_only[[1L]]
  if (!"correct_reward" %in% names(dat) || all(!is.finite(as_num(dat$correct_reward)))) dat$correct_reward <- meta$correct_reward[[1L]]
  if (!"pay_kl_on_stop" %in% names(dat)) dat$pay_kl_on_stop <- meta$pay_kl_on_stop[[1L]]
  if (!"input_type" %in% names(dat) || all(is.na(dat$input_type) | !nzchar(as.character(dat$input_type)))) dat$input_type <- meta$input_type[[1L]]
  if (!"duration_mode" %in% names(dat) || all(is.na(dat$duration_mode) | !nzchar(as.character(dat$duration_mode)))) dat$duration_mode <- meta$duration_mode[[1L]]
  dat$source_file <- path
  loaded[[length(loaded) + 1L]] <- dat
}
if (length(loaded) == 0L) stop("No usable evidence CSV rows were loaded.")
trial_data <- do.call(rbind, loaded)
message(sprintf("Loaded %d file(s) from %s.", length(loaded), input_dir))
message(sprintf("Loaded %d trial row(s).", nrow(trial_data)))

numeric_filter <- function(dat, col, value, label) {
  if (is.null(value) || !nzchar(trim_string(value)) || !col %in% names(dat)) return(dat)
  values <- as_num(parse_csv_values(value))
  keep <- rep(FALSE, nrow(dat))
  for (v in values) keep <- keep | parameter_equal(dat[[col]], v)
  out <- dat[keep, , drop = FALSE]
  message(sprintf("Filter %s=%s kept %d trial(s).", label, paste(values, collapse = ","), nrow(out)))
  out
}

string_filter <- function(dat, col, value, label) {
  if (is.null(value) || !nzchar(trim_string(value)) || !col %in% names(dat)) return(dat)
  values <- parse_csv_values(value)
  out <- dat[as.character(dat[[col]]) %in% values, , drop = FALSE]
  message(sprintf("Filter %s=%s kept %d trial(s).", label, paste(values, collapse = ","), nrow(out)))
  out
}

trial_data <- numeric_filter(trial_data, "loss_scale", loss_scale_arg, "loss_scale")
trial_data <- numeric_filter(trial_data, "alpha", alpha_arg, "alpha")
trial_data <- numeric_filter(trial_data, "seed", seeds_arg, "seed")
trial_data <- numeric_filter(trial_data, "rnn_units", rnn_units_arg, "rnn_units")
trial_data <- numeric_filter(trial_data, "latent_dim", latent_dim_arg, "latent_dim")
trial_data <- numeric_filter(trial_data, "correct_reward", correct_reward_arg, "correct_reward")
trial_data <- string_filter(trial_data, "input_type", input_type_arg, "input_type")
if (!is.na(pay_kl_on_stop_arg)) trial_data <- trial_data[as.logical(trial_data$pay_kl_on_stop) == isTRUE(pay_kl_on_stop_arg), , drop = FALSE]
if (!is.na(observer_only_arg)) trial_data <- trial_data[as.logical(trial_data$choice_at_end_only) == isTRUE(observer_only_arg), , drop = FALSE]
if (!is.null(checkpoints_arg) && nzchar(trim_string(checkpoints_arg))) {
  checkpoints <- parse_csv_values(checkpoints_arg)
  trial_data <- trial_data[as.character(trial_data$checkpoint) %in% checkpoints, , drop = FALSE]
}
if (nrow(trial_data) == 0L) stop("No trials remain after filters.")

trial_data$coherence_magnitude <- abs(as_num(trial_data$coherence))
if (!pool_coherence && length(simple_coherence_values) > 0L) {
  available_coh <- sort(unique(as_num(trial_data$coherence_magnitude)))
  available_coh <- available_coh[is.finite(available_coh)]
  snapped <- numeric()
  for (requested in simple_coherence_values) {
    if (length(available_coh) == 0L) next
    nearest <- available_coh[which.min(abs(available_coh - requested))]
    if (is.finite(nearest) && abs(nearest - requested) <= 1e-5) {
      snapped <- c(snapped, nearest)
    } else {
      warning(sprintf(
        "Requested simple coherence=%s was not found. Available: %s",
        num_label(requested),
        paste(vapply(available_coh, num_label, character(1)), collapse = ",")
      ))
    }
  }
  simple_coherence_values <- sort(unique(snapped))
  if (length(simple_coherence_values) == 0L) {
    stop("No requested --simple-coherence-values were found in the loaded trials.")
  }
  keep <- rep(FALSE, nrow(trial_data))
  for (value in simple_coherence_values) {
    keep <- keep | parameter_equal(trial_data$coherence_magnitude, value)
  }
  trial_data <- trial_data[keep, , drop = FALSE]
  selected_coherence <- NA_real_
  message(sprintf(
    "Using simple coherence magnitudes %s; %d trial(s) remain.",
    paste(vapply(simple_coherence_values, num_label, character(1)), collapse = ","),
    nrow(trial_data)
  ))
} else if (!pool_coherence) {
  if (!is.null(fixed_coherence_arg) && nzchar(trim_string(fixed_coherence_arg))) {
    selected_coherence <- as_num(fixed_coherence_arg)
  } else {
    possible <- as_num(trial_data$coherence_magnitude)
    nonzero <- possible[is.finite(possible) & possible > 0]
    selected_coherence <- if (length(nonzero) > 0L) mode_numeric(nonzero) else mode_numeric(possible)
  }
  trial_data <- trial_data[parameter_equal(trial_data$coherence_magnitude, selected_coherence), , drop = FALSE]
  message(sprintf("Using fixed coherence magnitude %s; %d trial(s) remain.", num_label(selected_coherence), nrow(trial_data)))
} else {
  selected_coherence <- NA_real_
  message("Pooling coherence magnitudes within each simplex fit.")
}
if (nrow(trial_data) == 0L) stop("No trials remain after coherence filter.")

fixed_opp <- if (!is.null(fixed_opp_arg) && nzchar(trim_string(fixed_opp_arg))) as_num(fixed_opp_arg) else 0
fixed_memory_lambda <- if (!is.null(fixed_memory_lambda_arg) && nzchar(trim_string(fixed_memory_lambda_arg))) {
  as_num(fixed_memory_lambda_arg)
} else if (!is.null(fixed_beta_arg) && nzchar(trim_string(fixed_beta_arg))) {
  as_num(fixed_beta_arg)
} else {
  mode_numeric(trial_data$memory_lambda)
}
memory_lambda_values <- as_num(parse_csv_values(memory_lambda_values_arg))
if (length(memory_lambda_values) == 0L) memory_lambda_values <- as_num(parse_csv_values(beta_values_arg))
if (length(memory_lambda_values) == 0L) {
  memory_lambda_values <- sort(unique(as_num(trial_data$memory_lambda[parameter_equal(trial_data$opportunity_cost, fixed_opp)])))
}
opportunity_values <- as_num(parse_csv_values(opportunity_values_arg))
if (length(opportunity_values) == 0L) {
  opportunity_values <- sort(unique(as_num(trial_data$opportunity_cost[parameter_equal(trial_data$memory_lambda, fixed_memory_lambda)])))
}
family_keep <- rep(FALSE, nrow(trial_data))
if (comparison_mode %in% c("both", "memory_lambda", "memory", "beta")) {
  mem_seed_values <- seed_values_for_family("beta")
  for (v in memory_lambda_values) {
    keep <- parameter_equal(trial_data$memory_lambda, v) & parameter_equal(trial_data$opportunity_cost, fixed_opp)
    keep <- keep & data_matches_seed_values(trial_data$seed, mem_seed_values)
    family_keep <- family_keep | keep
  }
}
if (comparison_mode %in% c("both", "opportunity")) {
  opp_seed_values <- seed_values_for_family("opportunity")
  for (v in opportunity_values) {
    keep <- parameter_equal(trial_data$opportunity_cost, v) & parameter_equal(trial_data$memory_lambda, fixed_memory_lambda)
    keep <- keep & data_matches_seed_values(trial_data$seed, opp_seed_values)
    family_keep <- family_keep | keep
  }
}
if (comparison_mode %in% c("checkpoint")) family_keep <- rep(TRUE, nrow(trial_data))
trial_data <- trial_data[family_keep, , drop = FALSE]
if (nrow(trial_data) == 0L) stop("No trials match requested memory-lambda/opportunity family settings.")

message(sprintf("Available run IDs: %s", paste(utils::head(sort(unique(as.character(trial_data$run_id))), 20L), collapse = ", ")))
message(sprintf("Available checkpoints: %s", paste(sort(unique(as.character(trial_data$checkpoint))), collapse = ",")))
message(sprintf("Available memory-lambda values: %s", paste(sort(unique(as_num(trial_data$memory_lambda))), collapse = ",")))
message(sprintf("Available opportunity costs: %s", paste(sort(unique(as_num(trial_data$opportunity_cost))), collapse = ",")))
message(sprintf("Available coherence magnitudes: %s", paste(sort(unique(as_num(trial_data$coherence_magnitude))), collapse = ",")))
message(sprintf("Available observation-noise std values: %s", paste(sort(unique(as_num(trial_data$observation_noise_std))), collapse = ",")))

if (!"training_step" %in% names(trial_data)) {
  trial_data$training_step <- -1
}
missing_training_step <- !is.finite(as_num(trial_data$training_step))
if (any(missing_training_step)) {
  message(sprintf(
    "Replacing %d missing training_step value(s) with -1 for grouping; checkpoint labels are retained.",
    sum(missing_training_step)
  ))
  trial_data$training_step[missing_training_step] <- -1
}

obs_mat_all <- as.matrix(trial_data[, obs_cols, drop = FALSE])
storage.mode(obs_mat_all) <- "double"
valid_obs <- rowSums(is.finite(obs_mat_all)) == max_observations
valid_count <- if ("num_observations" %in% names(trial_data)) as_num(trial_data$num_observations) == max_observations else rep(TRUE, nrow(trial_data))
valid_step_col <- paste0("valid_t", target_step)
valid_step <- if (valid_step_col %in% names(trial_data)) as.logical(trial_data[[valid_step_col]]) else rep(TRUE, nrow(trial_data))

a_col <- paste0("raw_logit_choose_a_t", target_step)
b_col <- paste0("raw_logit_choose_b_t", target_step)
logit_col <- paste0("choice_logit_t", target_step)
prob_col <- paste0("p_choose_b_given_terminal_t", target_step)
pa_col <- paste0("policy_choose_a_t", target_step)
pb_col <- paste0("policy_choose_b_t", target_step)
has_raw_logits <- all(c(a_col, b_col) %in% names(trial_data))
has_choice_logit <- logit_col %in% names(trial_data)
has_probability <- prob_col %in% names(trial_data) || all(c(pa_col, pb_col) %in% names(trial_data))

target_type <- target_type_arg
if (identical(target_type, "auto")) {
  target_type <- if (has_raw_logits || has_choice_logit) "logit" else if (has_probability) "probability" else "sampled_choice"
}
message(sprintf("Target type used: %s", target_type))

if (identical(target_type, "logit")) {
  if (has_raw_logits) {
    target <- as_num(trial_data[[b_col]]) - as_num(trial_data[[a_col]])
    if (has_choice_logit) {
      mismatch <- abs(target - as_num(trial_data[[logit_col]]))
      bad <- valid_step & is.finite(mismatch) & mismatch > 1e-4
      if (any(bad, na.rm = TRUE)) stop(sprintf("choice_logit_t%d does not match raw B-A logits.", target_step))
    }
  } else if (has_choice_logit) {
    target <- as_num(trial_data[[logit_col]])
  } else {
    stop("Requested target_type=logit, but final raw/choice logits are unavailable.")
  }
} else if (identical(target_type, "probability")) {
  if (prob_col %in% names(trial_data)) {
    target <- as_num(trial_data[[prob_col]])
  } else if (all(c(pa_col, pb_col) %in% names(trial_data))) {
    denom <- as_num(trial_data[[pa_col]]) + as_num(trial_data[[pb_col]])
    target <- as_num(trial_data[[pb_col]]) / denom
  } else {
    stop("Requested target_type=probability, but final terminal probabilities are unavailable.")
  }
} else {
  if (!"choose_right" %in% names(trial_data)) stop("Requested sampled_choice target but choose_right is unavailable.")
  target <- as.numeric(as.logical(trial_data$choose_right))
}
trial_data$target_value <- target

valid_target <- is.finite(target)
if (!identical(target_type, "logit")) valid_target <- valid_target & target >= 0 & target <= 1
trial_data <- trial_data[valid_obs & valid_count & valid_step & valid_target, , drop = FALSE]
if (nrow(trial_data) == 0L) stop("No valid trials remain for simplex fitting.")
message(sprintf("Included %d valid trial(s) with exactly %d observations.", nrow(trial_data), max_observations))

if (is.null(output_dir_arg) || !nzchar(trim_string(output_dir_arg))) {
  output_base_dir <- file.path(output_root, "evidence_accumulation_simplex_weights")
} else {
  output_base_dir <- output_dir_arg
}
observer_modes <- sort(unique(as.logical(trial_data$choice_at_end_only)))
observer_modes <- observer_modes[!is.na(observer_modes)]
observer_folder <- if (length(observer_modes) == 1L) {
  if (isTRUE(observer_modes[[1L]])) "observer_only" else "policy_timed"
} else {
  "observer_mixed"
}
coherence_folder <- if (pool_coherence) "coherence_pooled" else paste0("coherence_", value_token(selected_coherence))
if (simple_plot_mode) {
  simple_folder <- if (length(observer_modes) == 1L && isTRUE(observer_modes[[1L]])) {
    "observer_only_simple"
  } else if (length(observer_modes) == 1L && identical(observer_modes[[1L]], FALSE)) {
    "policy_timed_simple"
  } else {
    "observer_mixed_simple"
  }
  output_dir <- file.path(output_base_dir, simple_folder)
} else {
  output_dir <- file.path(output_base_dir, coherence_folder, observer_folder)
}
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
message(sprintf("Saving simplex outputs to %s.", output_dir))

group_cols <- c(
  "run_id", "checkpoint", "training_step", "seed", "loss_scale", "memory_lambda", "choice_at_end_only",
  "alpha", "beta", "opportunity_cost", "observation_noise_std"
)
if (!pool_coherence) group_cols <- c(group_cols, "coherence_magnitude")
group_cols <- group_cols[group_cols %in% names(trial_data)]

fit_group <- function(dat, group_values, group_index) {
  obs_mat <- as.matrix(dat[, obs_cols, drop = FALSE])
  storage.mode(obs_mat) <- "double"
  target <- as_num(dat$target_value)
  n <- nrow(dat)
  if (n < min_trials_per_fit) {
    return(list(run_rows = NULL, metrics = data.frame(group_values, n_trials = n, skipped_reason = sprintf("too_few_trials:%d", n)), predictions = NULL))
  }
  obs_var <- apply(obs_mat, 2, stats::var)
  if (any(!is.finite(obs_var) | obs_var <= 0)) {
    return(list(run_rows = NULL, metrics = data.frame(group_values, n_trials = n, skipped_reason = "zero_observation_variance"), predictions = NULL))
  }
  if (stats::var(target) <= 1e-12) {
    return(list(run_rows = NULL, metrics = data.frame(group_values, n_trials = n, skipped_reason = "near_zero_target_variance"), predictions = NULL))
  }
  fit <- fit_simplex_model(obs_mat, target, target_type, num_random_starts, analysis_seed + group_index * 997L)
  if (is.null(fit)) {
    return(list(run_rows = NULL, metrics = data.frame(group_values, n_trials = n, skipped_reason = "all_optimizer_starts_failed"), predictions = NULL))
  }
  pars <- extract_simplex_params(fit$par, max_observations)
  if (any(pars$weights < -1e-12) || abs(sum(pars$weights) - 1) > 1e-8 || pars$gain < -1e-12) {
    stop("Simplex parameter validation failed.")
  }
  ev <- evaluate_simplex(fit$par, obs_mat, target, target_type)
  cv <- cross_validate_simplex_model(obs_mat, target, target_type, num_cv_folds, max(2L, ceiling(num_random_starts / 2L)), analysis_seed + group_index * 2017L)
  uncon <- fit_unconstrained_model(obs_mat, target, target_type)
  uncon_coef <- if (is.null(uncon)) rep(NA_real_, max_observations) else uncon$coefficients
  negative_count <- if (is.null(uncon)) NA_integer_ else uncon$negative_count
  if (is.finite(negative_count) && negative_count > 0L) {
    warning(sprintf(
      "Unconstrained diagnostic found %d negative coefficient(s) for run_id=%s.",
      negative_count,
      as.character(group_values$run_id[[1L]])
    ))
  }
  middle <- 4:min(7, max_observations)
  edge <- unique(c(1, 2, max_observations - 1L, max_observations))
  bump <- mean(pars$weights[middle], na.rm = TRUE) - mean(pars$weights[edge], na.rm = TRUE)
  eff_bump <- mean(pars$effective[middle], na.rm = TRUE) - mean(pars$effective[edge], na.rm = TRUE)
  base <- as.data.frame(group_values, stringsAsFactors = FALSE)
  rows <- do.call(rbind, lapply(seq_len(max_observations), function(pos) {
    out <- base
    out$target_type <- target_type
    out$observation_position <- pos
    out$simplex_weight <- pars$weights[[pos]]
    out$gain <- pars$gain
    out$bias <- pars$bias
    out$effective_coefficient <- pars$effective[[pos]]
    out$unconstrained_coefficient <- uncon_coef[[pos]]
    out$n_trials <- n
    out$objective_value <- fit$value
    out$converged <- isTRUE(fit$convergence == 0L)
    out$optimizer_code <- fit$convergence
    out$optimizer_message <- fit$message
    out$num_iterations <- fit$counts
    out$num_random_starts <- num_random_starts
    out$finite_random_starts <- fit$n_finite_starts
    out$random_start_objective_sd <- fit$start_objective_sd
    out$full_data_R2 <- if (!is.null(ev$r2)) ev$r2 else NA_real_
    out$full_data_RMSE <- if (!is.null(ev$rmse)) ev$rmse else NA_real_
    out$full_data_MAE <- if (!is.null(ev$mae)) ev$mae else NA_real_
    out$full_data_loss <- ev$loss
    out$cross_validated_R2 <- cv$r2
    out$cross_validated_RMSE <- cv$rmse
    out$cross_validated_loss <- cv$loss
    out$unconstrained_R2 <- if (is.null(uncon)) NA_real_ else uncon$r2
    out$unconstrained_RMSE <- if (is.null(uncon)) NA_real_ else uncon$rmse
    out$negative_unconstrained_coefficients <- negative_count
    out$bump_contrast <- bump
    out$effective_bump_contrast <- eff_bump
    out
  }))
  metrics <- base
  metrics$target_type <- target_type
  metrics$n_trials <- n
  metrics$gain <- pars$gain
  metrics$bias <- pars$bias
  metrics$objective_value <- fit$value
  metrics$converged <- isTRUE(fit$convergence == 0L)
  metrics$optimizer_code <- fit$convergence
  metrics$optimizer_message <- fit$message
  metrics$num_iterations <- fit$counts
  metrics$num_random_starts <- num_random_starts
  metrics$finite_random_starts <- fit$n_finite_starts
  metrics$random_start_objective_sd <- fit$start_objective_sd
  metrics$full_data_R2 <- if (!is.null(ev$r2)) ev$r2 else NA_real_
  metrics$full_data_RMSE <- if (!is.null(ev$rmse)) ev$rmse else NA_real_
  metrics$full_data_MAE <- if (!is.null(ev$mae)) ev$mae else NA_real_
  metrics$full_data_loss <- ev$loss
  metrics$cross_validated_R2 <- cv$r2
  metrics$cross_validated_RMSE <- cv$rmse
  metrics$cross_validated_loss <- cv$loss
  metrics$unconstrained_R2 <- if (is.null(uncon)) NA_real_ else uncon$r2
  metrics$unconstrained_RMSE <- if (is.null(uncon)) NA_real_ else uncon$rmse
  metrics$negative_unconstrained_coefficients <- negative_count
  metrics$bump_contrast <- bump
  metrics$effective_bump_contrast <- eff_bump
  metrics$skipped_reason <- NA_character_
  predictions <- NULL
  if (save_predictions) {
    pred_logit <- predict_simplex_logit(fit$par, obs_mat)
    predictions <- data.frame(
      base[rep(1L, n), , drop = FALSE],
      trial_id = dat$trial_id,
      target_type = target_type,
      observed_final_logit = if (identical(target_type, "logit")) target else NA_real_,
      predicted_final_logit = pred_logit,
      observed_final_probability = if (identical(target_type, "logit")) stable_sigmoid(target) else target,
      predicted_final_probability = stable_sigmoid(pred_logit),
      stringsAsFactors = FALSE
    )
  }
  list(run_rows = rows, metrics = metrics, predictions = predictions)
}

split_key <- interaction(trial_data[, group_cols, drop = FALSE], drop = TRUE, lex.order = TRUE)
groups <- split(seq_len(nrow(trial_data)), split_key)
run_rows_list <- list()
metrics_list <- list()
pred_list <- list()
message(sprintf("Fitting %d independent run/condition group(s).", length(groups)))
for (idx in seq_along(groups)) {
  dat <- trial_data[groups[[idx]], , drop = FALSE]
  group_values <- dat[1L, group_cols, drop = FALSE]
  message(sprintf(
    "Fitting %d/%d: run_id=%s, seed=%s, memory_lambda=%s, opp=%s, obsstd=%s, n=%d",
    idx, length(groups), as.character(group_values$run_id[[1L]]), num_label(group_values$seed[[1L]]),
    num_label(group_values$memory_lambda[[1L]]), num_label(group_values$opportunity_cost[[1L]]),
    num_label(group_values$observation_noise_std[[1L]]), nrow(dat)
  ))
  res <- fit_group(dat, group_values, idx)
  if (!is.null(res$run_rows)) run_rows_list[[length(run_rows_list) + 1L]] <- res$run_rows
  if (!is.null(res$metrics)) metrics_list[[length(metrics_list) + 1L]] <- res$metrics
  if (!is.null(res$predictions)) pred_list[[length(pred_list) + 1L]] <- res$predictions
}
if (length(metrics_list) == 0L) stop("No simplex fit metrics were produced.")
fit_metrics <- rbind_fill(metrics_list)
if (!"skipped_reason" %in% names(fit_metrics)) fit_metrics$skipped_reason <- NA_character_
if (!"converged" %in% names(fit_metrics)) fit_metrics$converged <- NA
successful <- is.na(fit_metrics$skipped_reason) & fit_metrics$converged %in% TRUE
if (length(run_rows_list) == 0L) {
  utils::write.csv(fit_metrics, file.path(output_dir, "evidence_accumulation_simplex_fit_metrics.csv"), row.names = FALSE)
  stop("No successful simplex fits were produced.")
}
run_level <- rbind_fill(run_rows_list)
predictions <- rbind_fill(pred_list)
if (!"coherence_magnitude" %in% names(run_level)) run_level$coherence_magnitude <- NA_real_
if (!"coherence_magnitude" %in% names(fit_metrics)) fit_metrics$coherence_magnitude <- NA_real_

add_family_rows <- function(dat) {
  rows <- list()
  if (comparison_mode %in% c("both", "memory_lambda", "memory", "beta")) {
    keep <- parameter_equal(dat$opportunity_cost, fixed_opp)
    keep <- keep & data_matches_seed_values(dat$seed, seed_values_for_family("beta"))
    if (length(memory_lambda_values) > 0L) {
      keep_vals <- rep(FALSE, nrow(dat))
      for (v in memory_lambda_values) keep_vals <- keep_vals | parameter_equal(dat$memory_lambda, v)
      keep <- keep & keep_vals
    }
    if (any(keep)) {
      tmp <- dat[keep, , drop = FALSE]
      tmp$family <- "beta"
      tmp$parameter_value <- as_num(tmp$memory_lambda)
      rows[[length(rows) + 1L]] <- tmp
    }
  }
  if (comparison_mode %in% c("both", "opportunity")) {
    keep <- parameter_equal(dat$memory_lambda, fixed_memory_lambda)
    keep <- keep & data_matches_seed_values(dat$seed, seed_values_for_family("opportunity"))
    if (length(opportunity_values) > 0L) {
      keep_vals <- rep(FALSE, nrow(dat))
      for (v in opportunity_values) keep_vals <- keep_vals | parameter_equal(dat$opportunity_cost, v)
      keep <- keep & keep_vals
    }
    if (any(keep)) {
      tmp <- dat[keep, , drop = FALSE]
      tmp$family <- "opportunity"
      tmp$parameter_value <- as_num(tmp$opportunity_cost)
      rows[[length(rows) + 1L]] <- tmp
    }
  }
  if (comparison_mode %in% c("checkpoint")) {
    tmp <- dat
    tmp$family <- "checkpoint"
    tmp$parameter_value <- seq_len(nrow(tmp))
    rows[[length(rows) + 1L]] <- tmp
  }
  if (length(rows) == 0L) return(dat[FALSE, , drop = FALSE])
  do.call(rbind, rows)
}

run_level_family <- add_family_rows(run_level)
metrics_family <- add_family_rows(fit_metrics[is.na(fit_metrics$skipped_reason), , drop = FALSE])

summarize_one <- function(dat) {
  data.frame(
    n_runs = length(unique(dat$run_id)),
    mean_simplex_weight = mean(dat$simplex_weight, na.rm = TRUE),
    sd_simplex_weight = stats::sd(dat$simplex_weight, na.rm = TRUE),
    se_simplex_weight = sem_or_na(dat$simplex_weight),
    mean_gain = mean(dat$gain, na.rm = TRUE),
    sd_gain = stats::sd(dat$gain, na.rm = TRUE),
    se_gain = sem_or_na(dat$gain),
    mean_effective_coefficient = mean(dat$effective_coefficient, na.rm = TRUE),
    sd_effective_coefficient = stats::sd(dat$effective_coefficient, na.rm = TRUE),
    se_effective_coefficient = sem_or_na(dat$effective_coefficient),
    mean_unconstrained_coefficient = mean(dat$unconstrained_coefficient, na.rm = TRUE),
    sd_unconstrained_coefficient = stats::sd(dat$unconstrained_coefficient, na.rm = TRUE),
    se_unconstrained_coefficient = sem_or_na(dat$unconstrained_coefficient),
    mean_cross_validated_R2 = mean(dat$cross_validated_R2, na.rm = TRUE),
    se_cross_validated_R2 = sem_or_na(dat$cross_validated_R2)
  )
}

summary_group_cols <- c(
  "family", "parameter_value", "checkpoint", "training_step", "memory_lambda", "beta", "opportunity_cost",
  "coherence_magnitude", "observation_noise_std", "target_type", "observation_position"
)
summary_group_cols <- summary_group_cols[summary_group_cols %in% names(run_level_family)]
summary_key <- interaction(run_level_family[, summary_group_cols, drop = FALSE], drop = TRUE, lex.order = TRUE)
summary_rows <- lapply(split(seq_len(nrow(run_level_family)), summary_key), function(ii) {
  base <- run_level_family[ii[1L], summary_group_cols, drop = FALSE]
  cbind(base, summarize_one(run_level_family[ii, , drop = FALSE]))
})
summary_data <- if (length(summary_rows) > 0L) do.call(rbind, summary_rows) else data.frame()

run_path <- file.path(output_dir, "evidence_accumulation_simplex_weights_run_level.csv")
summary_path <- file.path(output_dir, "evidence_accumulation_simplex_weights_summary.csv")
metrics_path <- file.path(output_dir, "evidence_accumulation_simplex_fit_metrics.csv")
pred_path <- file.path(output_dir, "evidence_accumulation_simplex_predictions.csv")
utils::write.csv(run_level, run_path, row.names = FALSE)
utils::write.csv(summary_data, summary_path, row.names = FALSE)
utils::write.csv(fit_metrics, metrics_path, row.names = FALSE)
if (save_predictions && nrow(predictions) > 0L) utils::write.csv(predictions, pred_path, row.names = FALSE)

message(sprintf("Optimizer convergence rate: %d/%d successful fits.", sum(successful, na.rm = TRUE), nrow(fit_metrics)))
message(sprintf("Failed/skipped fits: %d.", sum(!is.na(fit_metrics$skipped_reason))))
message(sprintf("Simplex weight range: [%s, %s].", num_label(min(run_level$simplex_weight, na.rm = TRUE)), num_label(max(run_level$simplex_weight, na.rm = TRUE))))
weight_sum_dev <- aggregate(simplex_weight ~ run_id + checkpoint + seed + memory_lambda + opportunity_cost + observation_noise_std, run_level, sum)
message(sprintf("Maximum deviation from sum(weight)=1: %.3g.", max(abs(weight_sum_dev$simplex_weight - 1), na.rm = TRUE)))
message(sprintf("Gain range: [%s, %s].", num_label(min(fit_metrics$gain, na.rm = TRUE)), num_label(max(fit_metrics$gain, na.rm = TRUE))))
message(sprintf("Bias range: [%s, %s].", num_label(min(fit_metrics$bias, na.rm = TRUE)), num_label(max(fit_metrics$bias, na.rm = TRUE))))
message(sprintf("Negative unconstrained coefficients total: %d.", sum(run_level$unconstrained_coefficient < 0, na.rm = TRUE)))

target_panel_side_mm <- 40
target_panel_side_in <- target_panel_side_mm / 25.4
panel_margin_in <- c(bottom = 0.34, left = 0.45, top = 0.10, right = 0.10)
label_col_width_in <- 0.42
header_row_height_in <- 0.42

draw_row_label <- function(label) {
  plot.new()
  text(0.5, 0.5, label, srt = 90, cex = 1)
}

draw_header <- function(title, params, colors) {
  plot.new()
  text(0.5, 0.72, title, font = 2, cex = 0.95)
  if (length(params) == 0L) return(invisible(NULL))
  x0 <- seq(0.12, 0.88, length.out = length(params))
  points(x0, rep(0.34, length(params)), pch = 16, col = colors[as.character(params)], cex = 0.8)
  text(x0, rep(0.14, length(params)), labels = vapply(params, num_label, character(1)), cex = 0.75)
}

draw_error_bars <- function(x, y, se, col) {
  ok <- is.finite(x) & is.finite(y) & is.finite(se) & se > 0
  if (!any(ok)) return(invisible(NULL))
  graphics::arrows(x[ok], y[ok] - se[ok], x[ok], y[ok] + se[ok], angle = 90, code = 3, length = 0.025, col = col, lwd = 0.7)
}

plot_profile_panel <- function(dat, family, y_col, se_col, ylab, colors, params, ylim, reference = NULL) {
  plot(NA, NA, xlim = c(0.5, max_observations + 0.5), ylim = ylim, xlab = "Observation\nposition", ylab = ylab, xaxt = "n", las = 1)
  axis(1, at = seq_len(max_observations))
  if (is.finite(reference)) abline(h = reference, col = "grey70", lty = 2, lwd = 0.8)
  panel <- dat[dat$family == family, , drop = FALSE]
  for (param in params) {
    line <- panel[parameter_equal(panel$parameter_value, param), , drop = FALSE]
    if (nrow(line) == 0L) next
    line <- line[order(line$observation_position), , drop = FALSE]
    col <- colors[[as.character(param)]]
    lines(line$observation_position, line[[y_col]], col = col, lwd = 1.1)
    points(line$observation_position, line[[y_col]], col = col, pch = if (family == "beta") 16 else 17, cex = 0.65)
    if (se_col %in% names(line) && max(line$n_runs, na.rm = TRUE) > 1L) {
      draw_error_bars(line$observation_position, line[[y_col]], line[[se_col]], col)
    }
  }
}

plot_point_panel <- function(dat, family, y_col, ylab, colors, params, ylim) {
  plot(NA, NA, xlim = c(0.5, length(params) + 0.5), ylim = ylim, xlab = "", ylab = ylab, xaxt = "n", las = 1)
  axis(1, at = seq_along(params), labels = vapply(params, num_label, character(1)))
  panel <- dat[dat$family == family, , drop = FALSE]
  for (i in seq_along(params)) {
    param <- params[[i]]
    line <- panel[parameter_equal(panel$parameter_value, param), , drop = FALSE]
    if (nrow(line) == 0L) next
    y <- mean(line[[y_col]], na.rm = TRUE)
    se <- sem_or_na(line[[y_col]])
    col <- colors[[as.character(param)]]
    points(i, y, col = col, pch = if (family == "beta") 16 else 17, cex = 0.85)
    draw_error_bars(i, y, se, col)
  }
}

family_label <- function(family) {
  switch(
    family,
    beta = sprintf("Varying memory lambda\nopp = %s", num_label(fixed_opp)),
    opportunity = sprintf("Varying opportunity\nmemory lambda = %s", num_label(fixed_memory_lambda)),
    checkpoint = "Varying checkpoint",
    family
  )
}

family_params_for_plot <- function(family, plot_data, metric_data) {
  if (identical(family, "beta")) {
    return(sort(unique(as_num(memory_lambda_values))))
  }
  if (identical(family, "opportunity")) {
    return(sort(unique(as_num(opportunity_values))))
  }
  sort(unique(as_num(c(plot_data$parameter_value, metric_data$parameter_value))))
}

row_coherence_levels <- function(plot_data, metric_data) {
  values <- c(plot_data$coherence_magnitude, metric_data$coherence_magnitude)
  values <- sort(unique(as_num(values)))
  values[is.finite(values)]
}

coherence_keep <- function(dat, coh) {
  if (!"coherence_magnitude" %in% names(dat)) return(rep(TRUE, nrow(dat)))
  if (is.finite(as_num(coh))) parameter_equal(dat$coherence_magnitude, coh) else is.na(dat$coherence_magnitude)
}

plot_profile_empty <- function(ylab, ylim, reference = NULL) {
  plot(NA, NA, xlim = c(0.5, max_observations + 0.5), ylim = ylim, xlab = "Observation\nposition", ylab = ylab, xaxt = "n", las = 1)
  axis(1, at = seq_len(max_observations))
  if (is.finite(reference)) abline(h = reference, col = "grey70", lty = 2, lwd = 0.8)
  text(mean(par("usr")[1:2]), mean(par("usr")[3:4]), "No data", cex = 0.8, col = "grey40")
}

plot_point_empty <- function(params, ylab, ylim) {
  plot(NA, NA, xlim = c(0.5, max(1, length(params)) + 0.5), ylim = ylim, xlab = "", ylab = ylab, xaxt = "n", las = 1)
  if (length(params) > 0L) axis(1, at = seq_along(params), labels = vapply(params, num_label, character(1)))
  text(mean(par("usr")[1:2]), mean(par("usr")[3:4]), "No data", cex = 0.8, col = "grey40")
}

make_simplex_metric_plot <- function(plot_data, metric_data, spec, out_file, obsstd) {
  if (nrow(plot_data) == 0L && nrow(metric_data) == 0L) return(invisible(NULL))
  families <- unique(c(as.character(plot_data$family), as.character(metric_data$family)))
  families <- families[families %in% c("beta", "opportunity", "checkpoint")]
  if (length(families) == 0L) return(invisible(NULL))
  families <- intersect(c("beta", "opportunity", "checkpoint"), families)
  row_levels <- row_coherence_levels(plot_data, metric_data)
  if (length(row_levels) == 0L) row_levels <- NA_real_
  family_params <- setNames(vector("list", length(families)), families)
  family_colors <- setNames(vector("list", length(families)), families)
  for (family in families) {
    params <- family_params_for_plot(family, plot_data, metric_data)
    family_params[[family]] <- params
    family_colors[[family]] <- family_color_values(if (identical(family, "opportunity")) "opportunity" else "beta", params)
  }
  y_values <- if (identical(spec$type, "profile")) {
    c(
      plot_data[[spec$y_col]],
      plot_data[[spec$y_col]] - plot_data[[spec$se_col]],
      plot_data[[spec$y_col]] + plot_data[[spec$se_col]]
    )
  } else {
    metric_data[[spec$y_col]]
  }
  ylim <- safe_range(y_values, fallback = spec$fallback)
  panel_cell_width_in <- target_panel_side_in + panel_margin_in[["left"]] + panel_margin_in[["right"]]
  panel_cell_height_in <- target_panel_side_in + panel_margin_in[["bottom"]] + panel_margin_in[["top"]]
  layout_matrix <- matrix(0L, nrow = length(row_levels) + 1L, ncol = length(families) + 1L)
  next_id <- 1L
  for (j in seq_along(families)) {
    layout_matrix[1L, j + 1L] <- next_id
    next_id <- next_id + 1L
  }
  for (i in seq_along(row_levels)) {
    layout_matrix[i + 1L, 1L] <- next_id
    next_id <- next_id + 1L
    for (j in seq_along(families)) {
      layout_matrix[i + 1L, j + 1L] <- next_id
      next_id <- next_id + 1L
    }
  }
  png(
    out_file,
    width = label_col_width_in + length(families) * panel_cell_width_in,
    height = header_row_height_in + length(row_levels) * panel_cell_height_in,
    units = "in",
    res = 300,
    pointsize = 7,
    bg = "white"
  )
  layout(
    layout_matrix,
    widths = c(label_col_width_in / panel_cell_width_in, rep(1, length(families))),
    heights = c(header_row_height_in / panel_cell_height_in, rep(1, length(row_levels)))
  )
  old_par <- par(no.readonly = TRUE)
  par(cex = 1, cex.axis = 1, cex.lab = 1, oma = c(0, 0, 0, 0), xpd = FALSE)
  for (family in families) {
    par(mai = c(0.02, 0.02, 0.02, 0.04))
    draw_header(family_label(family), family_params[[family]], family_colors[[family]])
  }
  for (coh in row_levels) {
    par(mai = c(0.15, 0.02, 0.02, 0.02))
    draw_row_label(sprintf("coherence\n%s", if (is.finite(as_num(coh))) num_label(coh) else "pooled"))
    for (family in families) {
      params <- family_params[[family]]
      colors <- family_colors[[family]]
      if (identical(spec$type, "profile")) {
        row_data <- plot_data[
          coherence_keep(plot_data, coh) &
            parameter_equal(plot_data$observation_noise_std, obsstd) &
            plot_data$family == family,
          ,
          drop = FALSE
        ]
        par(mai = panel_margin_in)
        if (nrow(row_data) == 0L) {
          plot_profile_empty(spec$ylab, ylim, spec$reference)
        } else {
          plot_profile_panel(row_data, family, spec$y_col, spec$se_col, spec$ylab, colors, params, ylim, spec$reference)
        }
      } else {
        row_data <- metric_data[
          coherence_keep(metric_data, coh) &
            parameter_equal(metric_data$observation_noise_std, obsstd) &
            metric_data$family == family,
          ,
          drop = FALSE
        ]
        par(mai = panel_margin_in)
        if (nrow(row_data) == 0L) {
          plot_point_empty(params, spec$ylab, ylim)
        } else {
          plot_point_panel(row_data, family, spec$y_col, spec$ylab, colors, params, ylim)
        }
      }
    }
  }
  invisible(try(par(old_par), silent = TRUE))
  dev.off()
  message(sprintf("Saved %s", out_file))
  invisible(out_file)
}

make_simplex_weight_plot <- function(plot_data, metric_data, out_file) {
  if (nrow(plot_data) == 0L) return(invisible(NULL))
  beta_params <- sort(unique(as_num(memory_lambda_values)))
  opp_params <- sort(unique(as_num(opportunity_values)))
  beta_colors <- family_color_values("beta", beta_params)
  opp_colors <- family_color_values("opportunity", opp_params)
  layout_matrix <- matrix(seq_len(10L), nrow = 5L, byrow = TRUE)
  panel_cell_width_in <- target_panel_side_in + panel_margin_in[["left"]] + panel_margin_in[["right"]]
  panel_cell_height_in <- target_panel_side_in + panel_margin_in[["bottom"]] + panel_margin_in[["top"]]
  png(out_file, width = label_col_width_in + 2 * panel_cell_width_in, height = header_row_height_in + 4 * panel_cell_height_in,
      units = "in", res = 300, pointsize = 7, bg = "white")
  layout(
    rbind(c(0, 1, 2), c(3, 4, 5), c(6, 7, 8), c(9, 10, 11), c(12, 13, 14)),
    widths = c(label_col_width_in / panel_cell_width_in, 1, 1),
    heights = c(header_row_height_in / panel_cell_height_in, rep(1, 4))
  )
  old_par <- par(no.readonly = TRUE)
  par(cex = 1, cex.axis = 1, cex.lab = 1, oma = c(0, 0, 0, 0), xpd = FALSE)
  par(mai = c(0.02, 0.02, 0.02, 0.04))
  draw_header(sprintf("Varying memory lambda\nopp = %s", num_label(fixed_opp)), beta_params, beta_colors)
  par(mai = c(0.02, 0.02, 0.02, 0.04))
  draw_header(sprintf("Varying opportunity\nmemory lambda = %s", num_label(fixed_memory_lambda)), opp_params, opp_colors)
  ylim_w <- safe_range(c(plot_data$mean_simplex_weight, plot_data$mean_simplex_weight - plot_data$se_simplex_weight, plot_data$mean_simplex_weight + plot_data$se_simplex_weight), fallback = c(0, 0.2))
  ylim_eff <- safe_range(c(plot_data$mean_effective_coefficient, plot_data$mean_effective_coefficient - plot_data$se_effective_coefficient, plot_data$mean_effective_coefficient + plot_data$se_effective_coefficient), fallback = c(0, 1))
  ylim_gain <- safe_range(metric_data$gain, fallback = c(0, 1))
  ylim_cv <- safe_range(metric_data$cross_validated_R2, fallback = c(0, 1))
  rows <- list(
    list(label = "Relative\ntemporal\nweight", panel = function(fam, cols, params) plot_profile_panel(plot_data, fam, "mean_simplex_weight", "se_simplex_weight", "Relative\ntemporal weight", cols, params, ylim_w, 1 / max_observations)),
    list(label = "Effective\nevidence\ncoefficient", panel = function(fam, cols, params) plot_profile_panel(plot_data, fam, "mean_effective_coefficient", "se_effective_coefficient", "Effective\nevidence coefficient", cols, params, ylim_eff, NA_real_)),
    list(label = "Overall\nevidence\ngain", panel = function(fam, cols, params) plot_point_panel(metric_data, fam, "gain", "Overall\nevidence gain", cols, params, ylim_gain)),
    list(label = "CV fit\nquality", panel = function(fam, cols, params) plot_point_panel(metric_data, fam, "cross_validated_R2", "CV R2", cols, params, ylim_cv))
  )
  for (row in rows) {
    par(mai = c(0.15, 0.02, 0.02, 0.02))
    draw_row_label(row$label)
    par(mai = panel_margin_in)
    row$panel("beta", beta_colors, beta_params)
    par(mai = panel_margin_in)
    row$panel("opportunity", opp_colors, opp_params)
  }
  invisible(try(par(old_par), silent = TRUE))
  dev.off()
  message(sprintf("Saved %s", out_file))
  invisible(out_file)
}

if (!"coherence_magnitude" %in% names(summary_data)) summary_data$coherence_magnitude <- NA_real_
if (!"coherence_magnitude" %in% names(metrics_family)) metrics_family$coherence_magnitude <- NA_real_
plot_obsstd_levels <- sort(unique(as_num(c(summary_data$observation_noise_std, metrics_family$observation_noise_std))))
plot_obsstd_levels <- plot_obsstd_levels[is.finite(plot_obsstd_levels)]
plot_specs <- list(
  list(
    slug = "relative_temporal_weights",
    type = "profile",
    y_col = "mean_simplex_weight",
    se_col = "se_simplex_weight",
    ylab = "Relative\ntemporal weight",
    reference = 1 / max_observations,
    fallback = c(0, 0.2)
  ),
  list(
    slug = "effective_evidence_coefficients",
    type = "profile",
    y_col = "mean_effective_coefficient",
    se_col = "se_effective_coefficient",
    ylab = "Effective\nevidence coefficient",
    reference = NA_real_,
    fallback = c(0, 1)
  ),
  list(
    slug = "overall_evidence_gain",
    type = "point",
    y_col = "gain",
    ylab = "Overall\nevidence gain",
    reference = NA_real_,
    fallback = c(0, 1)
  ),
  list(
    slug = "cross_validated_fit_quality",
    type = "point",
    y_col = "cross_validated_R2",
    ylab = "CV R2",
    reference = NA_real_,
    fallback = c(0, 1)
  )
)
figure_paths <- character()
for (obsstd in plot_obsstd_levels) {
  subset_summary <- summary_data[parameter_equal(summary_data$observation_noise_std, obsstd), , drop = FALSE]
  subset_metrics <- metrics_family[parameter_equal(metrics_family$observation_noise_std, obsstd), , drop = FALSE]
  obs_suffix <- paste0("_obsstd_", value_token(obsstd))
  for (spec in plot_specs) {
    out_file <- file.path(output_dir, paste0("evidence_accumulation_simplex_", spec$slug, obs_suffix, ".png"))
    figure_paths <- c(figure_paths, make_simplex_metric_plot(subset_summary, subset_metrics, spec, out_file, obsstd))
  }
}

message(sprintf("Saved %s", run_path))
message(sprintf("Saved %s", summary_path))
message(sprintf("Saved %s", metrics_path))
if (save_predictions && nrow(predictions) > 0L) message(sprintf("Saved %s", pred_path))
message("Simplex model: weights = softmax(centered theta), gain = softplus(g_raw), prediction = bias + gain * weighted evidence.")
message("Simplex weights are relative allocation; gain is total evidence sensitivity; gain * weight_t is the effective coefficient.")
