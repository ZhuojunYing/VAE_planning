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
    "  Rscript analyses/exp_binary/plot_evidence_accumulation_integration_weights.R [evidence] [options]\n\n",
    "Fits psychophysical kernels from wide CSVs written by\n",
    "model_jax/evidence_accumulation_fixed_duration.py.\n\n",
    "Options:\n",
    "  --preset-file PATH          Preset CSV path. Default: analyses/exp_binary/evidence_accumulation_plot_presets.csv.\n",
    "                              Command-line values override preset values.\n",
    "  --input-dir DIR             CSV directory. Default: outputs/jax_simulations_evi_fixed_duration.\n",
    "  --use-training-simulations  Read regular post-training simulation CSVs instead of fixed-duration wide CSVs.\n",
    "                              Regular evidence_sample_t* columns are mapped to observation_*.\n",
    "  --output-root DIR           Output root. Default: results.\n",
    "  --output-dir DIR            Exact output directory. Default: results/evidence_accumulation_integration_weights.\n",
    "  --vary-memory-lambda-values LIST\n",
    "                              Memory-lambda values for the memory-vary column.\n",
    "                              Aliases: --memory-lambda-values, --memory-lambdas.\n",
    "  --vary-beta-values LIST     Legacy alias for --vary-memory-lambda-values when plotting older grids.\n",
    "                              Aliases: --beta-values, --betas.\n",
    "  --vary-opportunity-values LIST\n",
    "                              Opportunity costs for the opportunity-vary column.\n",
    "                              Aliases: --opportunity-values, --opportunities, --opportunity-costs.\n",
    "  --fixed-opp VALUE           Opportunity held fixed for beta-vary curves.\n",
    "                              Alias: --fixed-opportunity, --fixed-opportunity-cost.\n",
    "  --fixed-memory-lambda VALUE Memory-lambda held fixed for opportunity-vary curves.\n",
    "  --fixed-beta VALUE          Legacy alias for --fixed-memory-lambda.\n",
    "  --fixed-coherence VALUE     Coherence magnitude for fits. Alias: --coherence.\n",
    "                              If omitted, uses the most common nonzero coherence, or 0 if only 0 exists.\n",
    "  --pool-coherence            Pool all coherence magnitudes in the same regression.\n",
    "  --observation-noise-std LIST\n",
    "                              Observation noise std value(s) to plot. Aliases: --obsstd, --sigma.\n",
    "                              If omitted, uses all values in the selected fixed-duration CSVs.\n",
    "  --loss-scale VALUE          Filter loss scale. Aliases: --lambda, --lambda-value.\n",
    "  --correct-reward VALUE      Filter correct terminal reward scale. Default comes from preset when present.\n",
    "  --input-type VALUE          Filter trailing input type. Default comes from preset when present.\n",
    "  --pay-kl-on-stop            Use CSVs with the _stop_paid filename suffix. Default comes from preset when present.\n",
    "  --no-pay-kl-on-stop         Use legacy CSVs without the _stop_paid filename suffix.\n",
    "  --observer-only             Use observer/end-choice checkpoints only.\n",
    "  --non-observer              Use non-observer/self-timed checkpoints only.\n",
    "  --alpha VALUE               Filter alpha. Default: no filter.\n",
    "  --seeds LIST                Filter seed values. Default: all available.\n",
    "  --checkpoints LIST          Filter checkpoint labels. Default: all available.\n",
    "  --rnn-units VALUE           Filter RNN units if filename metadata is available. Default: no filter.\n",
    "  --latent-dim VALUE          Filter latent dimension if filename metadata is available. Default: no filter.\n",
    "  --max-observations VALUE    Number of fixed observations. Alias: --maxobs. Default: 10.\n",
    "  --min-trials-per-fit N      Minimum trials per independent glm fit. Default: 1000.\n",
    "  --extreme-coef VALUE        Mark coefficients with abs(raw) above this as unstable. Default: 50.\n",
    "  --stable-only               Plot/summarize only fits with no glm warnings, convergence failures,\n",
    "                              or extreme coefficients. Default: include finite ordinary-glm estimates\n",
    "                              and flag unstable fits in the CSV.\n",
    "  --integration-weights-only  Skip timecourse and choice-logit summaries/plots.\n",
    "  --skip-timecourse           Skip cumulative-evidence/KL/latent-delta timecourse summaries.\n",
    "  --skip-choice-logit         Skip choice-aligned logit timecourse summaries.\n",
    "  --simple-fixed-obsstd VALUE Plot only raw/standardized/normalized kernels for one obs std.\n",
    "                              Aliases: --simple-obsstd, --fixed-std-simple, --fixed-obsstd-simple.\n",
    "  --simple-coherence-values LIST\n",
    "                              Coherence magnitudes to show as rows in simple plots.\n",
    "                              Alias: --simple-coherences.\n",
    "  --simple-output-subdir NAME  Simple output folder. Default: observer_only_simple or no_observer_simple.\n",
    "  --no-cache                  Disable trial-data and summary caches.\n",
    "  --refresh-cache             Rebuild caches even when matching cache files exist.\n",
    "  --run-tests                 Run synthetic logistic-regression validation and exit.\n",
    "  --help                      Show this message.\n\n",
    "Examples:\n",
    "  Rscript analyses/exp_binary/plot_evidence_accumulation_integration_weights.R evidence \\\n",
    "    --vary-memory-lambda-values \"0,0.1,10\" \\\n",
    "    --vary-opportunity-values \"0.001,0.005,0.01\" \\\n",
    "    --fixed-coherence 0.2 \\\n",
    "    --observation-noise-std \"0.1,0.5,1.0\"\n",
    sep = ""
  )
}

if (any(args %in% c("--help", "-h"))) {
  usage()
  quit(save = "no", status = 0L)
}

trim_string <- function(value) trimws(as.character(value))

parse_csv_values <- function(value) {
  if (is.null(value) || !nzchar(trim_string(value))) {
    return(character())
  }
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

values_token <- function(values) {
  values <- sort(unique(as_num(values)))
  values <- values[is.finite(values)]
  paste(vapply(values, value_token, character(1)), collapse = "_")
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

has_any_option <- function(command_args, option_names) {
  for (arg in command_args) {
    for (option_name in option_names) {
      if (identical(arg, option_name) || startsWith(arg, paste0(option_name, "="))) {
        return(TRUE)
      }
    }
  }
  FALSE
}

extract_boolean_option <- function(args, true_names, default = FALSE) {
  value <- default
  keep <- rep(TRUE, length(args))
  i <- 1L
  while (i <= length(args)) {
    if (args[[i]] %in% true_names) {
      value <- TRUE
      keep[[i]] <- FALSE
    }
    i <- i + 1L
  }
  list(args = args[keep], value = value)
}

parse_bool_value <- function(value, default = FALSE, label = "boolean value") {
  if (is.null(value) || length(value) == 0L || is.na(value) || !nzchar(trim_string(value))) {
    return(default)
  }
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

opt <- extract_named_option(args, c("--observation-noise-std", "--obsstd", "--sigma"), NULL)
args <- opt$args
obsstd_arg <- opt$value

opt <- extract_named_option(args, c("--loss-scale", "--loss-scale-value", "--lambda", "--lambda-value"), NULL)
args <- opt$args
loss_scale_arg <- opt$value

opt <- extract_named_option(args, c("--correct-reward", "--reward-scale", "--terminal-correct-reward"), NULL)
args <- opt$args
correct_reward_arg <- opt$value

opt <- extract_named_option(args, c("--input-type"), NULL)
args <- opt$args
input_type_arg <- opt$value

opt <- extract_named_option(args, c("--alpha"), NULL)
args <- opt$args
alpha_arg <- opt$value

opt <- extract_named_option(args, c("--seeds"), NULL)
args <- opt$args
seeds_arg <- opt$value

opt <- extract_named_option(args, c("--checkpoints"), NULL)
args <- opt$args
checkpoints_arg <- opt$value

opt <- extract_named_option(args, c("--rnn-units", "--rnn-dims", "--rnn-dim"), NULL)
args <- opt$args
rnn_units_arg <- opt$value

opt <- extract_named_option(args, c("--latent-dim", "--latent-dims"), NULL)
args <- opt$args
latent_dim_arg <- opt$value

opt <- extract_named_option(args, c("--max-observations", "--max-observations-before-stop", "--maxobs"), NULL)
args <- opt$args
max_observations_arg <- opt$value

opt <- extract_named_option(args, c("--min-trials-per-fit"), "1000")
args <- opt$args
min_trials_per_fit <- as.integer(as_num(opt$value))

opt <- extract_named_option(args, c("--extreme-coef"), "50")
args <- opt$args
extreme_coef <- as_num(opt$value)

opt <- extract_boolean_option(args, c("--stable-only"), FALSE)
args <- opt$args
stable_only <- opt$value

opt <- extract_boolean_option(args, c("--integration-weights-only", "--weights-only"), FALSE)
args <- opt$args
integration_weights_only <- opt$value

opt <- extract_named_option(
  args,
  c("--simple-fixed-obsstd", "--simple-obsstd", "--fixed-std-simple", "--fixed-obsstd-simple"),
  NULL
)
args <- opt$args
simple_fixed_obsstd_arg <- opt$value

opt <- extract_named_option(args, c("--simple-coherence-values", "--simple-coherences"), NULL)
args <- opt$args
simple_coherence_values_arg <- opt$value

opt <- extract_named_option(args, c("--simple-output-subdir", "--simple-folder"), NULL)
args <- opt$args
simple_output_subdir_arg <- opt$value

opt <- extract_boolean_option(args, c("--skip-timecourse", "--skip-timecourse-summary"), FALSE)
args <- opt$args
skip_timecourse <- opt$value

opt <- extract_boolean_option(args, c("--skip-choice-logit", "--skip-logit-summary"), FALSE)
args <- opt$args
skip_choice_logit <- opt$value

if (integration_weights_only) {
  skip_timecourse <- TRUE
  skip_choice_logit <- TRUE
}

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
  sort(unique(vals[is.finite(vals)]))
} else {
  numeric()
}
simple_kernel_mode <- is.finite(simple_fixed_obsstd)
if (simple_kernel_mode) {
  skip_timecourse <- TRUE
  skip_choice_logit <- TRUE
  if (is.null(obsstd_arg) || !nzchar(trim_string(obsstd_arg))) {
    obsstd_arg <- as.character(simple_fixed_obsstd)
  }
}

opt <- extract_boolean_option(args, c("--no-cache"), FALSE)
args <- opt$args
use_cache <- !isTRUE(opt$value)

opt <- extract_boolean_option(args, c("--refresh-cache", "--rebuild-cache"), FALSE)
args <- opt$args
refresh_cache <- opt$value

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

opt <- extract_boolean_option(args, c("--run-tests"), FALSE)
args <- opt$args
run_tests <- opt$value

positional <- args[!startsWith(args, "--")]
task_name <- if (length(positional) > 0L) positional[[1L]] else "evidence"
if (!identical(task_name, "evidence")) {
  stop(sprintf("Only the evidence task is supported, not: %s", task_name))
}
extra_args <- setdiff(args, positional)
if (length(extra_args) > 0L) {
  stop(sprintf("Unexpected argument(s): %s", paste(extra_args, collapse = " ")))
}

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

preset_rows <- load_preset_rows(preset_file_arg, task_name)
preset_memory_row <- preset_rows$memory
preset_opp_row <- preset_rows$opportunity

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
if (!has_any_option(raw_command_args, c("--observation-noise-std", "--obsstd", "--sigma"))) {
  obsstd_arg <- preset_value(preset_memory_row, "observation_noise_std_arg", obsstd_arg)
}
if (!has_any_option(raw_command_args, c("--loss-scale", "--loss-scale-value", "--lambda", "--lambda-value"))) {
  loss_scale_arg <- preset_value(preset_memory_row, "loss_scale_arg", preset_value(preset_memory_row, "lambda_arg", loss_scale_arg))
}
if (!has_any_option(raw_command_args, c("--alpha"))) {
  alpha_arg <- preset_value(preset_memory_row, "alpha_arg", alpha_arg)
}
if (!has_any_option(raw_command_args, c("--seeds"))) {
  seeds_arg <- join_csv_unique(
    preset_value(preset_memory_row, "seed_arg", NULL),
    preset_value(preset_opp_row, "seed_arg", NULL)
  )
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
if (!pool_coherence && !has_any_option(raw_command_args, c("--fixed-coherence", "--coherence", "--pool-coherence"))) {
  preset_coherence_values <- parse_csv_values(preset_value(preset_memory_row, "coherence_arg", ""))
  if (length(preset_coherence_values) == 1L) {
    fixed_coherence_arg <- preset_coherence_values[[1L]]
  }
}
if (is.null(max_observations_arg) || !nzchar(trim_string(max_observations_arg))) {
  max_observations_arg <- "10"
}
max_observations <- as.integer(as_num(max_observations_arg))
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
  source_format <- if (grepl("_wide\\.csv$", base, perl = TRUE)) "duration_wide" else "training_sim"
  data.frame(
    source_file = path,
    source_format = source_format,
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

run_synthetic_test <- function() {
  set.seed(42)
  n <- 5000L
  n_obs <- 10L
  obs <- matrix(rnorm(n * n_obs), nrow = n)
  true_w <- seq(-1, 1, length.out = n_obs)
  p <- plogis(as.vector(obs %*% true_w))
  y <- rbinom(n, 1L, p)
  dat <- as.data.frame(obs)
  names(dat) <- paste0("observation_", seq_len(n_obs))
  dat$choose_right <- y
  fit <- stats::glm(
    stats::as.formula(paste("choose_right ~", paste(names(dat)[seq_len(n_obs)], collapse = " + "))),
    family = stats::binomial(link = "logit"),
    data = dat
  )
  est <- stats::coef(fit)[paste0("observation_", seq_len(n_obs))]
  if (stats::cor(true_w, est) < 0.98) {
    stop("Synthetic logistic-regression validation failed.")
  }
  message("Synthetic logistic-regression validation passed.")
}

if (run_tests) {
  run_synthetic_test()
  quit(save = "no", status = 0L)
}

if (!dir.exists(input_dir)) {
  stop(sprintf("Input directory does not exist: %s", input_dir))
}

if (is.null(output_dir_arg) || !nzchar(trim_string(output_dir_arg))) {
  output_base_dir <- file.path(output_root, "evidence_accumulation_integration_weights")
} else {
  output_base_dir <- output_dir_arg
}
cache_root <- file.path(output_base_dir, "_cache", "integration_weights")
if (use_cache) {
  dir.create(cache_root, recursive = TRUE, showWarnings = FALSE)
}

hash_text <- function(text) {
  path <- tempfile("integration_weight_cache_key_")
  on.exit(unlink(path), add = TRUE)
  writeLines(text, path, useBytes = TRUE)
  unname(tools::md5sum(path))
}

cache_relevant_args <- raw_command_args[!raw_command_args %in% c("--no-cache", "--refresh-cache", "--rebuild-cache")]

numeric_arg_values <- function(value) {
  values <- as_num_token(parse_csv_values(value))
  values[is.finite(values)]
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
if (length(prefilter_memory_values) == 0L) {
  prefilter_memory_values <- numeric_arg_values(beta_values_arg)
}
prefilter_opp_values <- numeric_arg_values(opportunity_values_arg)
prefilter_fixed_opp <- numeric_arg_values(fixed_opp_arg)
prefilter_fixed_memory <- numeric_arg_values(fixed_memory_lambda_arg)
if (length(prefilter_fixed_memory) == 0L) {
  prefilter_fixed_memory <- numeric_arg_values(fixed_beta_arg)
}
family_filters <- list()
if (length(prefilter_memory_values) > 0L && length(prefilter_fixed_opp) > 0L) {
  family_filters[[length(family_filters) + 1L]] <-
    metadata_matches_values(manifest$memory_lambda, prefilter_memory_values) &
    metadata_matches_values(manifest$opportunity_cost, prefilter_fixed_opp)
} else if (length(prefilter_memory_values) > 0L) {
  family_filters[[length(family_filters) + 1L]] <- metadata_matches_values(manifest$memory_lambda, prefilter_memory_values)
}
if (length(prefilter_opp_values) > 0L && length(prefilter_fixed_memory) > 0L) {
  family_filters[[length(family_filters) + 1L]] <-
    metadata_matches_values(manifest$opportunity_cost, prefilter_opp_values) &
    metadata_matches_values(manifest$memory_lambda, prefilter_fixed_memory)
} else if (length(prefilter_opp_values) > 0L) {
  family_filters[[length(family_filters) + 1L]] <- metadata_matches_values(manifest$opportunity_cost, prefilter_opp_values)
}
if (length(family_filters) > 0L) {
  family_keep <- Reduce(`|`, family_filters)
  prefilter_keep <- prefilter_keep & family_keep
}

selected_manifest <- manifest[prefilter_keep, , drop = FALSE]
message(sprintf(
  "Filename/metadata prefilter kept %d/%d wide CSV file(s) before reading rows.",
  nrow(selected_manifest),
  nrow(manifest)
))
empty_manifest <- manifest[!is.finite(manifest$file_size) | manifest$file_size <= 0, , drop = FALSE]
if (nrow(empty_manifest) > 0L) {
  warning(sprintf("Skipping %d empty or unreadable duration-controlled CSV file(s).", nrow(empty_manifest)))
}
if (nrow(selected_manifest) == 0L) {
  stop("No duration-controlled wide CSVs remain after filename/metadata filters.")
}
all_files <- selected_manifest$source_file

input_signature <- paste(
  c(
    "integration_weights_cache_v5",
    paste0("use_training_simulations=", use_training_simulations),
    normalizePath(input_dir, mustWork = FALSE),
    paste(cache_relevant_args, collapse = "\t"),
    paste(
      c(
        memory_lambda_values_arg = memory_lambda_values_arg,
        beta_values_arg = beta_values_arg,
        opportunity_values_arg = opportunity_values_arg,
        fixed_opp_arg = fixed_opp_arg,
        fixed_memory_lambda_arg = fixed_memory_lambda_arg,
        fixed_beta_arg = fixed_beta_arg,
        fixed_coherence_arg = fixed_coherence_arg,
        pool_coherence = pool_coherence,
        obsstd_arg = obsstd_arg,
        loss_scale_arg = loss_scale_arg,
        alpha_arg = alpha_arg,
        seeds_arg = seeds_arg,
        checkpoints_arg = checkpoints_arg,
        rnn_units_arg = rnn_units_arg,
        latent_dim_arg = latent_dim_arg,
        max_observations_arg = max_observations_arg,
        correct_reward_arg = correct_reward_arg,
        input_type_arg = input_type_arg,
        pay_kl_on_stop_arg = pay_kl_on_stop_arg,
        observer_only_arg = observer_only_arg,
        min_trials_per_fit = min_trials_per_fit,
        extreme_coef = extreme_coef,
        stable_only = stable_only,
        use_training_simulations = use_training_simulations
      ),
      collapse = "\t"
    ),
    paste(
      basename(selected_manifest$source_file),
      selected_manifest$file_size,
      selected_manifest$file_mtime,
      sep = "|",
      collapse = "\n"
    )
  ),
  collapse = "\n"
)
cache_key <- hash_text(input_signature)
trial_data_cache_path <- file.path(cache_root, paste0("trial_data_", cache_key, ".rds"))
fit_cache_path <- file.path(cache_root, paste0("glm_fits_", cache_key, ".rds"))
timecourse_cache_path <- file.path(cache_root, paste0("timecourse_summary_", cache_key, ".rds"))
summary_cache_path <- file.path(cache_root, paste0("integration_summary_", cache_key, ".rds"))
logit_cache_path <- file.path(cache_root, paste0("choice_logit_summary_", cache_key, ".rds"))
message(sprintf("Integration-weight cache key: %s", cache_key))

obs_cols <- paste0("observation_", seq_len(max_observations))
evidence_sample_cols <- paste0("evidence_sample_t", seq_len(max_observations))
required_cols <- c("trial_id", "run_id", "seed", "checkpoint", "training_step", "alpha", "beta", "opportunity_cost",
                   "coherence", "signed_coherence", "observation_noise_std", "correct_choice",
                   "correct_action", "terminal_action", "choose_right", "correct", "terminal_reward",
                   "total_opportunity_cost", "total_return", "num_observations", obs_cols,
                   evidence_sample_cols, "graph", "choose_correct", "total_reward")
metadata_optional_cols <- c(
  "loss_scale", "lambda", "memory_lambda", "choice_at_end_only", "duration_mode",
  "correct_reward", "pay_kl_on_stop", "input_type", "source_format"
)
action_cols <- unlist(lapply(seq_len(max_observations), function(t) {
  c(
    paste0("valid_t", t),
    paste0("action_t", t),
    paste0("stop_t", t),
    paste0("cumulative_evidence_t", t),
    paste0("kl_d_t", t),
    paste0("kl_d_obs_t", t),
    paste0("raw_logit_choose_a_t", t),
    paste0("raw_logit_choose_b_t", t),
    paste0("choice_logit_t", t),
    paste0("p_choose_b_given_terminal_t", t)
  )
}), use.names = FALSE)

loaded_file_count <- NA_integer_
trial_data_cache_hit <- FALSE
if (use_cache && !refresh_cache && file.exists(trial_data_cache_path)) {
  cached <- readRDS(trial_data_cache_path)
  trial_data <- cached$trial_data
  loaded_file_count <- cached$loaded_file_count
  trial_data_cache_hit <- TRUE
  message(sprintf(
    "Loaded cached filtered trial_data with %d row(s) from %s.",
    nrow(trial_data),
    trial_data_cache_path
  ))
} else {
  loaded <- list()
  for (path in all_files) {
    cols <- read_csv_names(path)
    latent_cols <- grep("^(z_mu|z_logvar|z_sigma|prior_mu|prior_logvar|prior_sigma)_[0-9]+_t[0-9]+$", cols, value = TRUE)
    select <- intersect(c(required_cols, metadata_optional_cols, action_cols, latent_cols), cols)
    has_observation_cols <- all(obs_cols %in% cols)
    has_evidence_sample_cols <- all(evidence_sample_cols %in% cols)
    if (!has_observation_cols && !has_evidence_sample_cols) {
      warning(sprintf(
        "Skipping %s because it lacks observation_1..observation_%d and evidence_sample_t1..t%d.",
        basename(path),
        max_observations,
        max_observations
      ))
      next
    }
    dat <- read_csv_fast(path, select = select)
    meta <- parse_model_metadata(path)
    if (!all(obs_cols %in% names(dat)) && all(evidence_sample_cols %in% names(dat))) {
      for (t in seq_len(max_observations)) {
        dat[[paste0("observation_", t)]] <- dat[[paste0("evidence_sample_t", t)]]
      }
    }
    if (!"trial_id" %in% names(dat)) {
      dat$trial_id <- if ("graph" %in% names(dat)) dat$graph else seq_len(nrow(dat))
    }
    if (!"run_id" %in% names(dat)) {
      dat$run_id <- sub("\\.csv$", "", basename(path))
    }
    if (!"checkpoint" %in% names(dat) || all(is.na(dat$checkpoint) | !nzchar(as.character(dat$checkpoint)))) {
      dat$checkpoint <- meta$checkpoint[[1L]]
    }
    if (!"training_step" %in% names(dat)) {
      dat$training_step <- NA_real_
    }
    if (!"correct" %in% names(dat) && "choose_correct" %in% names(dat)) {
      dat$correct <- dat$choose_correct
    }
    if (!"total_return" %in% names(dat) && "total_reward" %in% names(dat)) {
      dat$total_return <- dat$total_reward
    }
    for (nm in names(meta)) {
      if (identical(nm, "source_file")) next
      if (!nm %in% names(dat) || all(!is.finite(as_num(dat[[nm]])))) {
        dat[[nm]] <- meta[[nm]][[1L]]
      }
    }
    if (!"loss_scale" %in% names(dat) || all(!is.finite(as_num(dat$loss_scale)))) {
      dat$loss_scale <- if ("lambda" %in% names(dat)) as_num(dat$lambda) else meta$loss_scale[[1L]]
    }
    if (!"lambda" %in% names(dat) || all(!is.finite(as_num(dat$lambda)))) {
      dat$lambda <- as_num(dat$loss_scale)
    }
    if (!"memory_lambda" %in% names(dat) || all(!is.finite(as_num(dat$memory_lambda)))) {
      dat$memory_lambda <- meta$memory_lambda[[1L]]
    }
    if (!"choice_at_end_only" %in% names(dat)) {
      dat$choice_at_end_only <- meta$choice_at_end_only[[1L]]
    }
    if (!"correct_reward" %in% names(dat) || all(!is.finite(as_num(dat$correct_reward)))) {
      dat$correct_reward <- meta$correct_reward[[1L]]
    }
    if (!"pay_kl_on_stop" %in% names(dat)) {
      dat$pay_kl_on_stop <- meta$pay_kl_on_stop[[1L]]
    }
    if (!"input_type" %in% names(dat) || all(is.na(dat$input_type) | !nzchar(as.character(dat$input_type)))) {
      dat$input_type <- meta$input_type[[1L]]
    }
    if (!"duration_mode" %in% names(dat) || all(is.na(dat$duration_mode) | !nzchar(as.character(dat$duration_mode)))) {
      dat$duration_mode <- meta$duration_mode[[1L]]
    }
    if (!"source_format" %in% names(dat) || all(is.na(dat$source_format) | !nzchar(as.character(dat$source_format)))) {
      dat$source_format <- meta$source_format[[1L]]
    }
    dat$source_file <- path
    loaded[[length(loaded) + 1L]] <- dat
  }

  if (length(loaded) == 0L) {
    stop(sprintf("No usable %s CSVs were loaded.", if (use_training_simulations) "regular evidence simulation" else "duration-controlled wide"))
  }
  trial_data <- do.call(rbind, loaded)
  loaded_file_count <- length(loaded)
  message(sprintf(
    "Loaded %d %s file(s) from %s.",
    loaded_file_count,
    if (use_training_simulations) "regular evidence simulation" else "duration-controlled",
    input_dir
  ))
  message(sprintf("Loaded %d trial row(s).", nrow(trial_data)))
}

numeric_filter <- function(dat, col, value, label) {
  if (is.null(value) || !nzchar(trim_string(value))) return(dat)
  values <- as_num(parse_csv_values(value))
  keep <- rep(FALSE, nrow(dat))
  for (v in values) keep <- keep | parameter_equal(dat[[col]], v)
  out <- dat[keep, , drop = FALSE]
  message(sprintf("Filter %s=%s kept %d trial(s).", label, paste(values, collapse = ","), nrow(out)))
  out
}

string_filter <- function(dat, col, value, label) {
  if (is.null(value) || !nzchar(trim_string(value))) return(dat)
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

if (!is.na(pay_kl_on_stop_arg)) {
  trial_data <- trial_data[as.logical(trial_data$pay_kl_on_stop) == isTRUE(pay_kl_on_stop_arg), , drop = FALSE]
  message(sprintf(
    "Filter pay_kl_on_stop=%s kept %d trial(s).",
    if (isTRUE(pay_kl_on_stop_arg)) "true" else "false",
    nrow(trial_data)
  ))
}

if (!is.na(observer_only_arg)) {
  trial_data <- trial_data[as.logical(trial_data$choice_at_end_only) == isTRUE(observer_only_arg), , drop = FALSE]
  message(sprintf(
    "Filter observer_only=%s kept %d trial(s).",
    if (isTRUE(observer_only_arg)) "true" else "false",
    nrow(trial_data)
  ))
}

if (!is.null(checkpoints_arg) && nzchar(trim_string(checkpoints_arg))) {
  checkpoints <- parse_csv_values(checkpoints_arg)
  trial_data <- trial_data[as.character(trial_data$checkpoint) %in% checkpoints, , drop = FALSE]
  message(sprintf("Filter checkpoint=%s kept %d trial(s).", paste(checkpoints, collapse = ","), nrow(trial_data)))
}

if (nrow(trial_data) == 0L) {
  stop("No trials remain after filters.")
}

if ("num_observations" %in% names(trial_data)) {
  fixed_rows <- as.logical(trial_data$choice_at_end_only)
  bad_n <- sum(fixed_rows & as_num(trial_data$num_observations) != max_observations, na.rm = TRUE)
  if (bad_n > 0L) {
    stop(sprintf("%d observer-only trial(s) do not have exactly %d observations.", bad_n, max_observations))
  }
  bad_policy_n <- sum(!fixed_rows & (as_num(trial_data$num_observations) < 1 | as_num(trial_data$num_observations) > max_observations), na.rm = TRUE)
  if (bad_policy_n > 0L) {
    stop(sprintf("%d self-timed trial(s) have invalid num_observations outside [1, %d].", bad_policy_n, max_observations))
  }
}
missing_obs <- setdiff(obs_cols, names(trial_data))
if (length(missing_obs) > 0L) stop(sprintf("Missing observation columns: %s", paste(missing_obs, collapse = ", ")))

for (t in seq_len(max_observations - 1L)) {
  action_col <- paste0("action_t", t)
  stop_col <- paste0("stop_t", t)
  fixed_rows <- as.logical(trial_data$choice_at_end_only)
  if (action_col %in% names(trial_data) && any(fixed_rows & as_num(trial_data[[action_col]]) != 0, na.rm = TRUE)) {
    stop(sprintf("Found non-CONTINUE action before final fixed-duration timestep t=%d.", t))
  }
  if (stop_col %in% names(trial_data) && any(fixed_rows & as.logical(trial_data[[stop_col]]), na.rm = TRUE)) {
    stop(sprintf("Found terminal action before final fixed-duration timestep t=%d.", t))
  }
}

for (t in seq_len(max_observations)) {
  a_col <- paste0("raw_logit_choose_a_t", t)
  b_col <- paste0("raw_logit_choose_b_t", t)
  d_col <- paste0("choice_logit_t", t)
  p_col <- paste0("p_choose_b_given_terminal_t", t)
  if (all(c(a_col, b_col, d_col) %in% names(trial_data))) {
    diff <- as_num(trial_data[[b_col]]) - as_num(trial_data[[a_col]])
    delta <- abs(diff - as_num(trial_data[[d_col]]))
    valid_col <- paste0("valid_t", t)
    valid_step <- if (valid_col %in% names(trial_data)) as.logical(trial_data[[valid_col]]) else rep(TRUE, nrow(trial_data))
    bad <- valid_step & is.finite(delta) & delta > 1e-4
    if (any(bad, na.rm = TRUE)) {
      stop(sprintf(
        "choice_logit_t%d does not match raw B-A logits; max absolute mismatch is %g.",
        t,
        max(delta[is.finite(delta)], na.rm = TRUE)
      ))
    }
  }
  if (p_col %in% names(trial_data)) {
    p <- as_num(trial_data[[p_col]])
    valid_col <- paste0("valid_t", t)
    valid_step <- if (valid_col %in% names(trial_data)) as.logical(trial_data[[valid_col]]) else rep(TRUE, nrow(trial_data))
    if (any(valid_step & (!is.finite(p) | p < -1e-8 | p > 1 + 1e-8), na.rm = TRUE)) {
      stop(sprintf("Invalid conditional A/B probability at t=%d.", t))
    }
  }
}

trial_data$choose_right_num <- as.integer(as.logical(trial_data$choose_right))
trial_data$coherence_magnitude <- abs(as_num(trial_data$coherence))

if (pool_coherence) {
  selected_coherence_label <- "pooled"
  message("Pooling coherence magnitudes within each regression.")
} else if (simple_kernel_mode && length(simple_coherence_values) > 0L) {
  available_coh <- sort(unique(as_num(trial_data$coherence_magnitude)))
  requested <- simple_coherence_values
  snapped <- numeric()
  for (value in requested) {
    diffs <- abs(available_coh - value)
    best_i <- which.min(diffs)
    if (length(best_i) == 0L || !is.finite(diffs[[best_i]]) || diffs[[best_i]] > 1e-5) {
      warning(sprintf(
        "Requested simple coherence=%s was not found. Available: %s",
        num_label(value),
        paste(vapply(available_coh, num_label, character(1)), collapse = ",")
      ))
      next
    }
    snapped <- c(snapped, available_coh[[best_i]])
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
  selected_coherence_label <- values_token(simple_coherence_values)
  message(sprintf(
    "Using simple coherence magnitudes %s; %d trial(s) remain.",
    paste(vapply(simple_coherence_values, num_label, character(1)), collapse = ","),
    nrow(trial_data)
  ))
} else {
  if (!is.null(fixed_coherence_arg) && nzchar(trim_string(fixed_coherence_arg))) {
    selected_coherence <- as_num(fixed_coherence_arg)
  } else {
    possible_coh <- as_num(trial_data$coherence_magnitude)
    nonzero <- possible_coh[is.finite(possible_coh) & abs(possible_coh) > 1e-12]
    selected_coherence <- if (length(nonzero) > 0L) mode_numeric(nonzero) else mode_numeric(possible_coh)
  }
  trial_data <- trial_data[parameter_equal(trial_data$coherence_magnitude, selected_coherence), , drop = FALSE]
  selected_coherence_label <- num_label(selected_coherence)
  message(sprintf("Using fixed coherence magnitude %s; %d trial(s) remain.", selected_coherence_label, nrow(trial_data)))
  if (simple_kernel_mode && length(simple_coherence_values) == 0L) {
    simple_coherence_values <- selected_coherence
  }
}

obsstd_values <- as_num(parse_csv_values(obsstd_arg))
if (length(obsstd_values) > 0L) {
  keep <- rep(FALSE, nrow(trial_data))
  for (v in obsstd_values) keep <- keep | parameter_equal(trial_data$observation_noise_std, v)
  trial_data <- trial_data[keep, , drop = FALSE]
} else {
  obsstd_values <- sort(unique(as_num(trial_data$observation_noise_std)))
  obsstd_values <- obsstd_values[is.finite(obsstd_values)]
}

if (nrow(trial_data) == 0L) {
  stop("No trials remain after coherence/noise filters.")
}

fixed_opp <- if (!is.null(fixed_opp_arg) && nzchar(trim_string(fixed_opp_arg))) as_num(fixed_opp_arg) else mode_numeric(trial_data$opportunity_cost)
fixed_memory_lambda <- if (!is.null(fixed_memory_lambda_arg) && nzchar(trim_string(fixed_memory_lambda_arg))) {
  as_num(fixed_memory_lambda_arg)
} else if (!is.null(fixed_beta_arg) && nzchar(trim_string(fixed_beta_arg))) {
  as_num(fixed_beta_arg)
} else {
  mode_numeric(trial_data$memory_lambda)
}

memory_lambda_values <- as_num(parse_csv_values(memory_lambda_values_arg))
if (length(memory_lambda_values) == 0L) {
  memory_lambda_values <- as_num(parse_csv_values(beta_values_arg))
}
if (length(memory_lambda_values) == 0L) {
  memory_lambda_values <- sort(unique(as_num(trial_data$memory_lambda[parameter_equal(trial_data$opportunity_cost, fixed_opp)])))
}
opportunity_values <- as_num(parse_csv_values(opportunity_values_arg))
if (length(opportunity_values) == 0L) {
  opportunity_values <- sort(unique(as_num(trial_data$opportunity_cost[parameter_equal(trial_data$memory_lambda, fixed_memory_lambda)])))
}

family_keep <- rep(FALSE, nrow(trial_data))
for (v in memory_lambda_values) {
  family_keep <- family_keep | (parameter_equal(trial_data$memory_lambda, v) & parameter_equal(trial_data$opportunity_cost, fixed_opp))
}
for (v in opportunity_values) {
  family_keep <- family_keep | (parameter_equal(trial_data$opportunity_cost, v) & parameter_equal(trial_data$memory_lambda, fixed_memory_lambda))
}
trial_data <- trial_data[family_keep, , drop = FALSE]
if (nrow(trial_data) == 0L) {
  stop("No trials match memory-lambda/opportunity comparison conditions.")
}

if ("training_step" %in% names(trial_data)) {
  training_step_num <- as_num(trial_data$training_step)
  missing_training_step <- !is.finite(training_step_num)
  if (any(missing_training_step)) {
    message(sprintf(
      "Replacing %d missing training_step value(s) with -1 for grouping; checkpoint labels are retained.",
      sum(missing_training_step)
    ))
    training_step_num[missing_training_step] <- -1
    trial_data$training_step <- training_step_num
  }
}

if (use_cache && !trial_data_cache_hit) {
  saveRDS(
    list(trial_data = trial_data, loaded_file_count = loaded_file_count),
    trial_data_cache_path
  )
  message(sprintf(
    "Cached filtered trial_data with %d row(s) to %s.",
    nrow(trial_data),
    trial_data_cache_path
  ))
}

message(sprintf("Available runs: %d", length(unique(trial_data$run_id))))
message(sprintf("Available checkpoints: %s", paste(sort(unique(as.character(trial_data$checkpoint))), collapse = ",")))
message(sprintf("Available beta values: %s", paste(sort(unique(as_num(trial_data$beta))), collapse = ",")))
message(sprintf("Available memory-lambda values: %s", paste(sort(unique(as_num(trial_data$memory_lambda))), collapse = ",")))
message(sprintf("Available opportunity costs: %s", paste(sort(unique(as_num(trial_data$opportunity_cost))), collapse = ",")))
message(sprintf("Available coherence magnitudes: %s", paste(sort(unique(as_num(trial_data$coherence_magnitude))), collapse = ",")))
message(sprintf("Available observation-noise std values: %s", paste(sort(unique(as_num(trial_data$observation_noise_std))), collapse = ",")))
message(sprintf("Available observer-only modes: %s", paste(sort(unique(as.character(as.logical(trial_data$choice_at_end_only)))), collapse = ",")))

if (is.null(output_dir_arg) || !nzchar(trim_string(output_dir_arg))) {
  output_base_dir <- file.path(output_root, "evidence_accumulation_integration_weights")
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
coherence_folder <- if (pool_coherence) {
  "coherence_pooled"
} else {
  paste0("coherence_", value_token(selected_coherence_label))
}
output_dir <- file.path(output_base_dir, coherence_folder, observer_folder)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
message(sprintf("Saving integration-weight plots to %s.", output_dir))

fit_group_cols <- c(
  "run_id", "checkpoint", "training_step", "seed", "loss_scale", "memory_lambda", "choice_at_end_only", "alpha", "beta", "opportunity_cost",
  "observation_noise_std", "num_observations"
)
if (!pool_coherence) {
  fit_group_cols <- c(fit_group_cols, "coherence_magnitude")
}
fit_group_cols <- fit_group_cols[fit_group_cols %in% names(trial_data)]

fit_one_kernel <- function(dat, group_values) {
  group_n_obs <- if ("num_observations" %in% names(dat)) {
    unique(as.integer(round(as_num(dat$num_observations))))
  } else {
    max_observations
  }
  group_n_obs <- group_n_obs[is.finite(group_n_obs)]
  if (length(group_n_obs) != 1L || group_n_obs < 1L || group_n_obs > max_observations) {
    return(list(rows = NULL, contrast = NULL, skipped = "invalid_num_observations_group"))
  }
  fit_obs_cols <- paste0("observation_", seq_len(group_n_obs))
  complete <- stats::complete.cases(dat[, c("choose_right_num", fit_obs_cols), drop = FALSE])
  dat <- dat[complete, , drop = FALSE]
  n <- nrow(dat)
  if (n < min_trials_per_fit) {
    return(list(rows = NULL, contrast = NULL, skipped = sprintf("too_few_trials:%d", n)))
  }
  if (length(unique(dat$choose_right_num)) < 2L) {
    return(list(rows = NULL, contrast = NULL, skipped = "single_terminal_choice"))
  }
  formula <- stats::as.formula(paste("choose_right_num ~", paste(fit_obs_cols, collapse = " + ")))
  warnings <- character()
  fit <- tryCatch(
    withCallingHandlers(
      stats::glm(formula, family = stats::binomial(link = "logit"), data = dat),
      warning = function(w) {
        warnings <<- c(warnings, conditionMessage(w))
        invokeRestart("muffleWarning")
      }
    ),
    error = function(e) e
  )
  if (inherits(fit, "error")) {
    return(list(rows = NULL, contrast = NULL, skipped = paste0("glm_error:", conditionMessage(fit))))
  }
  coef_table <- summary(fit)$coefficients
  if (!all(fit_obs_cols %in% rownames(coef_table))) {
    return(list(rows = NULL, contrast = NULL, skipped = "missing_glm_coefficients"))
  }
  raw <- coef_table[fit_obs_cols, "Estimate"]
  se <- coef_table[fit_obs_cols, "Std. Error"]
  obs_sd <- vapply(fit_obs_cols, function(col) stats::sd(as_num(dat[[col]]), na.rm = TRUE), numeric(1))
  denom <- sum(abs(raw), na.rm = TRUE)
  normalized <- if (is.finite(denom) && denom > 0) raw / denom else rep(NA_real_, length(raw))
  converged <- isTRUE(fit$converged) && length(warnings) == 0L && all(is.finite(raw)) && !any(abs(raw) > extreme_coef, na.rm = TRUE)

  base <- as.data.frame(group_values, stringsAsFactors = FALSE)
  rows <- do.call(rbind, lapply(seq_along(fit_obs_cols), function(i) {
    out <- base
    out$timestep <- i
    out$integration_weight_raw <- as.numeric(raw[[i]])
    out$standard_error <- as.numeric(se[[i]])
    out$ci_lower <- out$integration_weight_raw - 1.96 * out$standard_error
    out$ci_upper <- out$integration_weight_raw + 1.96 * out$standard_error
    out$integration_weight_standardized <- as.numeric(raw[[i]] * obs_sd[[i]])
    out$integration_weight_normalized <- as.numeric(normalized[[i]])
    out$observation_sd <- as.numeric(obs_sd[[i]])
    out$n_trials <- n
    out$converged <- converged
    out$glm_converged <- isTRUE(fit$converged)
    out$had_warning <- length(warnings) > 0L
    out$warning_message <- paste(unique(warnings), collapse = " | ")
    out$extreme_coefficient <- any(abs(raw) > extreme_coef, na.rm = TRUE)
    out
  }))
  contrast <- base
  contrast$n_trials <- n
  contrast$converged <- converged
  if (length(raw) >= 10L) {
    contrast$bump_contrast_raw <- mean(raw[4:7], na.rm = TRUE) - mean(raw[c(1, 2, 9, 10)], na.rm = TRUE)
    contrast$bump_contrast_standardized <- mean((raw * obs_sd)[4:7], na.rm = TRUE) -
      mean((raw * obs_sd)[c(1, 2, 9, 10)], na.rm = TRUE)
    contrast$bump_contrast_normalized <- mean(normalized[4:7], na.rm = TRUE) -
      mean(normalized[c(1, 2, 9, 10)], na.rm = TRUE)
  } else {
    contrast$bump_contrast_raw <- NA_real_
    contrast$bump_contrast_standardized <- NA_real_
    contrast$bump_contrast_normalized <- NA_real_
  }
  list(rows = rows, contrast = contrast, skipped = NULL)
}

fit_count <- NA_integer_
if (use_cache && !refresh_cache && file.exists(fit_cache_path)) {
  cached <- readRDS(fit_cache_path)
  weights <- cached$weights
  contrasts <- cached$contrasts
  skipped <- cached$skipped
  fit_count <- cached$fit_count
  message(sprintf("Loaded cached GLM fit outputs from %s.", fit_cache_path))
} else {
  split_key <- interaction(trial_data[, fit_group_cols, drop = FALSE], drop = TRUE, lex.order = TRUE)
  groups <- split(seq_len(nrow(trial_data)), split_key)
  fit_rows <- list()
  contrast_rows <- list()
  skipped <- data.frame(reason = character(), n_trials = integer(), stringsAsFactors = FALSE)

  for (idx in seq_along(groups)) {
    dat <- trial_data[groups[[idx]], , drop = FALSE]
    group_values <- dat[1L, fit_group_cols, drop = FALSE]
    res <- fit_one_kernel(dat, group_values)
    if (!is.null(res$rows)) fit_rows[[length(fit_rows) + 1L]] <- res$rows
    if (!is.null(res$contrast)) contrast_rows[[length(contrast_rows) + 1L]] <- res$contrast
    if (!is.null(res$skipped)) {
      skipped <- rbind(skipped, data.frame(reason = res$skipped, n_trials = nrow(dat), stringsAsFactors = FALSE))
    }
  }

  if (length(fit_rows) == 0L) {
    if (nrow(skipped) > 0L) {
      message("Skipped fits:")
      print(utils::head(skipped, 20L))
    }
    stop("No logistic-regression fits with finite coefficients were produced.")
  }

  weights <- do.call(rbind, fit_rows)
  contrasts <- if (length(contrast_rows) > 0L) do.call(rbind, contrast_rows) else data.frame()
  fit_count <- length(fit_rows)
  if (use_cache) {
    saveRDS(
      list(weights = weights, contrasts = contrasts, skipped = skipped, fit_count = fit_count),
      fit_cache_path
    )
    message(sprintf("Cached GLM fit outputs to %s.", fit_cache_path))
  }
}
message(sprintf("Produced %d fitted integration-weight row(s) from %d independent fit(s).", nrow(weights), fit_count))
if (nrow(skipped) > 0L) {
  message(sprintf("Skipped %d fit(s).", nrow(skipped)))
  print(utils::head(skipped, 20L))
}
message(sprintf(
  "Raw coefficient range: [%s, %s]",
  num_label(min(weights$integration_weight_raw, na.rm = TRUE)),
  num_label(max(weights$integration_weight_raw, na.rm = TRUE))
))
message(sprintf(
  "Stable ordinary-glm fits: %d/%d. %s",
  length(unique(weights$run_id[weights$converged])),
  length(unique(weights$run_id)),
  if (stable_only) "Plotting stable fits only." else "Plotting all finite ordinary-glm estimates; stability flags are saved in the CSV."
))

add_family_rows <- function(dat) {
  rows <- list()
  memory_keep <- parameter_equal(dat$opportunity_cost, fixed_opp)
  if (length(memory_lambda_values) > 0L) {
    keep <- rep(FALSE, nrow(dat))
    for (v in memory_lambda_values) keep <- keep | parameter_equal(dat$memory_lambda, v)
    memory_keep <- memory_keep & keep
  }
  if (any(memory_keep)) {
    tmp <- dat[memory_keep, , drop = FALSE]
    tmp$family <- "beta"
    tmp$parameter_value <- as_num(tmp$memory_lambda)
    rows[[length(rows) + 1L]] <- tmp
  }
  opp_keep <- parameter_equal(dat$memory_lambda, fixed_memory_lambda)
  if (length(opportunity_values) > 0L) {
    keep <- rep(FALSE, nrow(dat))
    for (v in opportunity_values) keep <- keep | parameter_equal(dat$opportunity_cost, v)
    opp_keep <- opp_keep & keep
  }
  if (any(opp_keep)) {
    tmp <- dat[opp_keep, , drop = FALSE]
    tmp$family <- "opportunity"
    tmp$parameter_value <- as_num(tmp$opportunity_cost)
    rows[[length(rows) + 1L]] <- tmp
  }
  if (length(rows) == 0L) return(dat[FALSE, , drop = FALSE])
  do.call(rbind, rows)
}

weights_family <- add_family_rows(weights)
contrasts_family <- if (nrow(contrasts) > 0L) add_family_rows(contrasts) else contrasts
if (nrow(weights_family) == 0L) stop("No fitted weights match requested memory-lambda/opportunity family settings.")

weights_path <- file.path(output_dir, "evidence_accumulation_integration_weights.csv")
summary_path <- file.path(output_dir, "evidence_accumulation_integration_weights_summary.csv")
contrast_path <- file.path(output_dir, "evidence_accumulation_integration_weight_bump_contrast.csv")
skipped_path <- file.path(output_dir, "evidence_accumulation_integration_weight_skipped_fits.csv")
timecourse_summary_path <- file.path(output_dir, "evidence_accumulation_fixed_duration_timecourse_summary.csv")
utils::write.csv(weights, weights_path, row.names = FALSE)
if (nrow(contrasts) > 0L) utils::write.csv(contrasts, contrast_path, row.names = FALSE)
if (nrow(skipped) > 0L) utils::write.csv(skipped, skipped_path, row.names = FALSE)

row_mean_or_na <- function(mat) {
  if (is.null(mat) || length(mat) == 0L) return(numeric())
  out <- rowMeans(mat, na.rm = TRUE)
  all_missing <- rowSums(is.finite(mat)) == 0L
  out[all_missing] <- NA_real_
  out
}

step_latent_matrix <- function(dat, prefix, step, transform = identity) {
  pattern <- sprintf("^%s_([0-9]+)_t%d$", prefix, step)
  cols <- grep(pattern, names(dat), value = TRUE)
  if (length(cols) == 0L) return(NULL)
  dims <- as.integer(sub(pattern, "\\1", cols))
  cols <- cols[order(dims)]
  mat <- as.matrix(dat[, cols, drop = FALSE])
  storage.mode(mat) <- "double"
  transform(mat)
}

same_latent_shape <- function(a, b, c = NULL) {
  if (is.null(a) || is.null(b)) return(FALSE)
  ok <- identical(dim(a), dim(b))
  if (!is.null(c)) ok <- ok && identical(dim(a), dim(c))
  ok
}

clip_exp_half <- function(mat) exp(0.5 * pmin(pmax(mat, -10), 10))

build_timecourse_long <- function(dat) {
  rows <- vector("list", max_observations)
  choice_direction <- ifelse(as.logical(dat$choose_right), 1, -1)
  prev_action_aligned_logit <- NULL
  prev_z_mu <- NULL
  prev_z_sigma <- NULL
  warned_missing_prior <- FALSE
  for (step in seq_len(max_observations)) {
    cumulative_col <- paste0("cumulative_evidence_t", step)
    kl_col <- paste0("kl_d_t", step)
    logit_col <- paste0("choice_logit_t", step)
    valid_col <- paste0("valid_t", step)
    valid_step <- if (valid_col %in% names(dat)) {
      as.logical(dat[[valid_col]])
    } else if ("num_observations" %in% names(dat)) {
      step <= as_num(dat$num_observations)
    } else {
      rep(TRUE, nrow(dat))
    }

    action_aligned_logit <- if (logit_col %in% names(dat)) {
      ifelse(valid_step, choice_direction * as_num(dat[[logit_col]]), NA_real_)
    } else {
      rep(NA_real_, nrow(dat))
    }
    abs_delta_action_logit <- if (!is.null(prev_action_aligned_logit)) {
      abs(action_aligned_logit - prev_action_aligned_logit)
    } else {
      rep(NA_real_, nrow(dat))
    }

    prior_mu <- step_latent_matrix(dat, "prior_mu", step)
    prior_sigma <- step_latent_matrix(dat, "prior_sigma", step)
    if (is.null(prior_sigma)) {
      prior_logvar <- step_latent_matrix(dat, "prior_logvar", step)
      if (!is.null(prior_logvar)) prior_sigma <- sqrt(exp(prior_logvar) + 1e-6)
    }

    z_mu_raw <- step_latent_matrix(dat, "z_mu", step)
    if (!is.null(z_mu_raw) && same_latent_shape(z_mu_raw, prior_mu, prior_sigma)) {
      z_mu <- (z_mu_raw - prior_mu) / pmax(prior_sigma, 1e-8)
    } else if (!is.null(z_mu_raw)) {
      z_mu <- z_mu_raw
      if (!warned_missing_prior) {
        warning("Prior columns are missing or mismatched; z_mu deltas use raw z_mu.")
        warned_missing_prior <- TRUE
      }
    } else {
      z_mu <- NULL
    }

    z_sigma_raw <- step_latent_matrix(dat, "z_sigma", step)
    if (is.null(z_sigma_raw)) {
      z_logvar <- step_latent_matrix(dat, "z_logvar", step, transform = clip_exp_half)
      z_sigma_raw <- z_logvar
    }
    if (!is.null(z_sigma_raw) && !is.null(prior_sigma) && identical(dim(z_sigma_raw), dim(prior_sigma))) {
      z_sigma <- z_sigma_raw / pmax(prior_sigma, 1e-8)
    } else if (!is.null(z_sigma_raw)) {
      z_sigma <- z_sigma_raw
      if (!warned_missing_prior) {
        warning("Prior columns are missing or mismatched; z_sigma deltas use raw z_sigma.")
        warned_missing_prior <- TRUE
      }
    } else {
      z_sigma <- NULL
    }

    abs_delta_z_mu <- if (!is.null(z_mu) && !is.null(prev_z_mu) && identical(dim(z_mu), dim(prev_z_mu))) {
      row_mean_or_na(abs(z_mu - prev_z_mu))
    } else {
      rep(NA_real_, nrow(dat))
    }
    signed_delta_z_sigma <- if (!is.null(z_sigma) && !is.null(prev_z_sigma) && identical(dim(z_sigma), dim(prev_z_sigma))) {
      row_mean_or_na(z_sigma - prev_z_sigma)
    } else {
      rep(NA_real_, nrow(dat))
    }

    rows[[step]] <- data.frame(
      run_id = dat$run_id,
      checkpoint = dat$checkpoint,
      training_step = as_num(dat$training_step),
      seed = as_num(dat$seed),
      loss_scale = as_num(dat$loss_scale),
      memory_lambda = as_num(dat$memory_lambda),
      choice_at_end_only = as.logical(dat$choice_at_end_only),
      alpha = as_num(dat$alpha),
      beta = as_num(dat$beta),
      opportunity_cost = as_num(dat$opportunity_cost),
      coherence_magnitude = dat$coherence_magnitude,
      observation_noise_std = as_num(dat$observation_noise_std),
      num_observations = as_num(dat$num_observations),
      timestep = step,
      choice_aligned_cumulative_evidence = if (cumulative_col %in% names(dat)) {
        ifelse(valid_step, choice_direction * as_num(dat[[cumulative_col]]), NA_real_)
      } else {
        rep(NA_real_, nrow(dat))
      },
      delta_kl = if (kl_col %in% names(dat)) ifelse(valid_step, as_num(dat[[kl_col]]), NA_real_) else rep(NA_real_, nrow(dat)),
      abs_delta_z_mu = ifelse(valid_step, abs_delta_z_mu, NA_real_),
      signed_delta_z_sigma = ifelse(valid_step, signed_delta_z_sigma, NA_real_),
      abs_delta_action_aligned_action_logit = abs_delta_action_logit,
      stringsAsFactors = FALSE
    )
    if (!is.null(action_aligned_logit)) prev_action_aligned_logit <- action_aligned_logit
    if (!is.null(z_mu)) prev_z_mu <- z_mu
    if (!is.null(z_sigma)) prev_z_sigma <- z_sigma
  }
  do.call(rbind, rows)
}

summarize_timecourse <- function(dat) {
  if (nrow(dat) == 0L) return(data.frame())
  family_dat <- add_family_rows(dat)
  if (nrow(family_dat) == 0L) return(data.frame())
  metrics <- c(
    choice_aligned_cumulative_evidence = "choice_aligned_cumulative_evidence",
    delta_kl = "delta_kl",
    abs_delta_z_mu = "abs_delta_z_mu",
    signed_delta_z_sigma = "signed_delta_z_sigma",
    abs_delta_action_aligned_action_logit = "abs_delta_action_aligned_action_logit"
  )
  run_cols <- c(
    "family", "parameter_value", "run_id", "checkpoint", "training_step",
    "loss_scale", "memory_lambda", "choice_at_end_only", "alpha", "beta", "opportunity_cost",
    "observation_noise_std", "num_observations", "timestep"
  )
  if (!pool_coherence) run_cols <- c(run_cols, "coherence_magnitude")
  out <- list()
  for (metric in names(metrics)) {
    metric_col <- metrics[[metric]]
    metric_dat <- family_dat[is.finite(as_num(family_dat[[metric_col]])), , drop = FALSE]
    if (nrow(metric_dat) == 0L) next
    run_mean <- aggregate(
      metric_dat[[metric_col]],
      by = metric_dat[, run_cols, drop = FALSE],
      FUN = mean,
      na.rm = TRUE
    )
    names(run_mean)[names(run_mean) == "x"] <- "run_value"
    run_n <- aggregate(
      metric_dat[[metric_col]],
      by = metric_dat[, run_cols, drop = FALSE],
      FUN = function(x) sum(is.finite(as_num(x)))
    )
    names(run_n)[names(run_n) == "x"] <- "n_trials"
    run_mean <- merge(run_mean, run_n, by = run_cols, all = TRUE)
    group_cols <- setdiff(run_cols, "run_id")
    mean_ag <- aggregate(run_mean$run_value, by = run_mean[, group_cols, drop = FALSE], FUN = mean, na.rm = TRUE)
    names(mean_ag)[names(mean_ag) == "x"] <- "mean_weight"
    sd_ag <- aggregate(run_mean$run_value, by = run_mean[, group_cols, drop = FALSE], FUN = stats::sd, na.rm = TRUE)
    names(sd_ag)[names(sd_ag) == "x"] <- "sd_across_runs"
    n_ag <- aggregate(run_mean$run_id, by = run_mean[, group_cols, drop = FALSE], FUN = function(x) length(unique(x)))
    names(n_ag)[names(n_ag) == "x"] <- "n_runs"
    ntr_ag <- aggregate(run_mean$n_trials, by = run_mean[, group_cols, drop = FALSE], FUN = sum, na.rm = TRUE)
    names(ntr_ag)[names(ntr_ag) == "x"] <- "n_trials"
    summary <- merge(mean_ag, sd_ag, by = group_cols, all = TRUE)
    summary <- merge(summary, n_ag, by = group_cols, all = TRUE)
    summary <- merge(summary, ntr_ag, by = group_cols, all = TRUE)
    summary$se_across_runs <- ifelse(summary$n_runs > 1L, summary$sd_across_runs / sqrt(summary$n_runs), NA_real_)
    summary$metric <- metric
    out[[length(out) + 1L]] <- summary
  }
  if (length(out) == 0L) return(data.frame())
  do.call(rbind, out)
}

if (skip_timecourse) {
  timecourse_summary <- data.frame()
  message("Skipping timecourse summaries because --skip-timecourse/--integration-weights-only was requested.")
} else if (use_cache && !refresh_cache && file.exists(timecourse_cache_path)) {
  timecourse_summary <- readRDS(timecourse_cache_path)
  message(sprintf("Loaded cached timecourse summary from %s.", timecourse_cache_path))
} else {
  timecourse_long <- build_timecourse_long(trial_data)
  timecourse_summary <- summarize_timecourse(timecourse_long)
  if (use_cache) {
    saveRDS(timecourse_summary, timecourse_cache_path)
    message(sprintf("Cached timecourse summary to %s.", timecourse_cache_path))
  }
}
if (nrow(timecourse_summary) > 0L) {
  utils::write.csv(timecourse_summary, timecourse_summary_path, row.names = FALSE)
}

metric_cols <- c(
  raw = "integration_weight_raw",
  standardized = "integration_weight_standardized",
  normalized = "integration_weight_normalized"
)

summarize_weights <- function(dat, metric_col) {
  dat <- dat[is.finite(as_num(dat[[metric_col]])), , drop = FALSE]
  if (nrow(dat) == 0L) {
    return(data.frame())
  }
  group_cols <- c("family", "parameter_value", "checkpoint", "training_step", "beta", "opportunity_cost",
                  "loss_scale", "memory_lambda", "choice_at_end_only", "alpha", "observation_noise_std",
                  "num_observations", "timestep")
  if ("coherence_magnitude" %in% names(dat)) group_cols <- c(group_cols, "coherence_magnitude")
  ag <- aggregate(dat[[metric_col]], by = dat[, group_cols, drop = FALSE], FUN = mean, na.rm = TRUE)
  names(ag)[names(ag) == "x"] <- "mean_weight"
  sd_ag <- aggregate(dat[[metric_col]], by = dat[, group_cols, drop = FALSE], FUN = stats::sd, na.rm = TRUE)
  names(sd_ag)[names(sd_ag) == "x"] <- "sd_across_runs"
  n_ag <- aggregate(dat$run_id, by = dat[, group_cols, drop = FALSE], FUN = function(x) length(unique(x)))
  names(n_ag)[names(n_ag) == "x"] <- "n_runs"
  ntr_ag <- aggregate(dat$n_trials, by = dat[, group_cols, drop = FALSE], FUN = sum, na.rm = TRUE)
  names(ntr_ag)[names(ntr_ag) == "x"] <- "n_trials"
  out <- merge(ag, sd_ag, by = group_cols, all = TRUE)
  out <- merge(out, n_ag, by = group_cols, all = TRUE)
  out <- merge(out, ntr_ag, by = group_cols, all = TRUE)
  out$se_across_runs <- ifelse(out$n_runs > 1L, out$sd_across_runs / sqrt(out$n_runs), NA_real_)
  out
}

if (use_cache && !refresh_cache && file.exists(summary_cache_path)) {
  summary_weights <- readRDS(summary_cache_path)
  message(sprintf("Loaded cached integration-weight summary from %s.", summary_cache_path))
} else {
  summary_list <- list()
  summary_source <- weights_family
  if (stable_only) {
    summary_source <- summary_source[summary_source$converged, , drop = FALSE]
    if (nrow(summary_source) == 0L) {
      stop("No stable ordinary-glm fits remain after --stable-only filtering.")
    }
  }
  for (metric_name in names(metric_cols)) {
    tmp <- summarize_weights(summary_source, metric_cols[[metric_name]])
    if (nrow(tmp) == 0L) next
    tmp$metric <- metric_name
    summary_list[[length(summary_list) + 1L]] <- tmp
  }
  summary_weights <- if (length(summary_list) > 0L) do.call(rbind, summary_list) else data.frame()
  if (use_cache) {
    saveRDS(summary_weights, summary_cache_path)
    message(sprintf("Cached integration-weight summary to %s.", summary_cache_path))
  }
}
utils::write.csv(summary_weights, summary_path, row.names = FALSE)

build_logit_long <- function(dat) {
  needed <- unlist(lapply(seq_len(max_observations), function(t) paste0("choice_logit_t", t)), use.names = FALSE)
  if (!all(needed %in% names(dat))) return(data.frame())
  rows <- vector("list", max_observations)
  choice_direction <- ifelse(as.logical(dat$choose_right), 1, -1)
  for (t in seq_len(max_observations)) {
    valid_col <- paste0("valid_t", t)
    valid_step <- if (valid_col %in% names(dat)) {
      as.logical(dat[[valid_col]])
    } else if ("num_observations" %in% names(dat)) {
      t <= as_num(dat$num_observations)
    } else {
      rep(TRUE, nrow(dat))
    }
    rows[[t]] <- data.frame(
      run_id = dat$run_id,
      checkpoint = dat$checkpoint,
      training_step = as_num(dat$training_step),
      seed = as_num(dat$seed),
      loss_scale = as_num(dat$loss_scale),
      memory_lambda = as_num(dat$memory_lambda),
      choice_at_end_only = as.logical(dat$choice_at_end_only),
      alpha = as_num(dat$alpha),
      beta = as_num(dat$beta),
      opportunity_cost = as_num(dat$opportunity_cost),
      coherence_magnitude = dat$coherence_magnitude,
      observation_noise_std = as_num(dat$observation_noise_std),
      num_observations = as_num(dat$num_observations),
      timestep = t,
      choice_aligned_choice_logit = ifelse(valid_step, choice_direction * as_num(dat[[paste0("choice_logit_t", t)]]), NA_real_),
      stringsAsFactors = FALSE
    )
  }
  do.call(rbind, rows)
}

if (skip_choice_logit) {
  logit_summary <- data.frame()
  message("Skipping choice-logit summaries because --skip-choice-logit/--integration-weights-only was requested.")
} else if (use_cache && !refresh_cache && file.exists(logit_cache_path)) {
  logit_summary <- readRDS(logit_cache_path)
  message(sprintf("Loaded cached choice-logit summary from %s.", logit_cache_path))
} else {
  logit_long <- build_logit_long(trial_data)
  logit_summary <- data.frame()
  if (nrow(logit_long) > 0L) {
    logit_family <- add_family_rows(logit_long)
    run_cols <- c("family", "parameter_value", "run_id", "checkpoint", "training_step", "loss_scale", "memory_lambda", "choice_at_end_only", "alpha", "beta", "opportunity_cost",
                  "observation_noise_std", "num_observations", "timestep")
    if (!pool_coherence) run_cols <- c(run_cols, "coherence_magnitude")
    run_mean <- aggregate(
      logit_family$choice_aligned_choice_logit,
      by = logit_family[, run_cols, drop = FALSE],
      FUN = mean,
      na.rm = TRUE
    )
    names(run_mean)[names(run_mean) == "x"] <- "value"
    group_cols <- setdiff(run_cols, "run_id")
    logit_summary <- aggregate(run_mean$value, by = run_mean[, group_cols, drop = FALSE], FUN = mean, na.rm = TRUE)
    names(logit_summary)[names(logit_summary) == "x"] <- "mean_weight"
    logit_sd <- aggregate(run_mean$value, by = run_mean[, group_cols, drop = FALSE], FUN = stats::sd, na.rm = TRUE)
    names(logit_sd)[names(logit_sd) == "x"] <- "sd_across_runs"
    logit_n <- aggregate(run_mean$run_id, by = run_mean[, group_cols, drop = FALSE], FUN = function(x) length(unique(x)))
    names(logit_n)[names(logit_n) == "x"] <- "n_runs"
    logit_summary <- merge(logit_summary, logit_sd, by = group_cols, all = TRUE)
    logit_summary <- merge(logit_summary, logit_n, by = group_cols, all = TRUE)
    logit_summary$se_across_runs <- ifelse(logit_summary$n_runs > 1L, logit_summary$sd_across_runs / sqrt(logit_summary$n_runs), NA_real_)
  }
  if (use_cache) {
    saveRDS(logit_summary, logit_cache_path)
    message(sprintf("Cached choice-logit summary to %s.", logit_cache_path))
  }
}
if (nrow(logit_summary) > 0L) {
  utils::write.csv(logit_summary, file.path(output_dir, "evidence_accumulation_choice_aligned_choice_logit_summary.csv"), row.names = FALSE)
}

target_panel_side_mm <- 33
target_panel_side_in <- target_panel_side_mm / 25.4
panel_margin_in <- c(bottom = 0.55, left = 0.67, top = 0.08, right = 0.10)
label_margin_in <- c(bottom = 0.02, left = 0.02, top = 0.02, right = 0.02)
label_col_width_in <- 0.42
header_row_height_in <- 0.72

positive_log_floor <- function(values, fallback = 1e-8) {
  values <- as_num(values)
  positive <- values[is.finite(values) & values > 0]
  if (length(positive) == 0L) return(fallback)
  max(10^floor(log10(min(positive))), fallback)
}

positive_log_axis_limit <- function(values, fallback = c(1e-8, 1)) {
  values <- as_num(values)
  positive <- values[is.finite(values) & values > 0]
  if (length(positive) == 0L) return(fallback)
  lower <- positive_log_floor(positive)
  upper <- max(positive)
  if (!is.finite(upper) || upper <= lower) upper <- lower * 10
  c(lower, upper)
}

signed_log_axis_limit <- function(values, fallback = c(-1, 1)) {
  values <- as_num(values)
  values <- values[is.finite(values)]
  if (length(values) == 0L) return(fallback)
  max_abs <- max(abs(values), na.rm = TRUE)
  if (!is.finite(max_abs) || max_abs <= 0) return(fallback)
  c(-max_abs, max_abs)
}

transform_y_values <- function(values, y_scale = "linear", floor_value = 1e-8) {
  values <- as_num(values)
  if (identical(y_scale, "log10")) {
    return(log10(pmax(values, floor_value)))
  }
  if (identical(y_scale, "signed_log10")) {
    return(sign(values) * log10(1 + abs(values) / floor_value))
  }
  values
}

log_axis_ticks <- function(raw_lim) {
  raw_lim <- as_num(raw_lim)
  lower <- max(min(raw_lim, na.rm = TRUE), 1e-300)
  upper <- max(raw_lim, na.rm = TRUE)
  if (!is.finite(lower) || !is.finite(upper) || upper <= 0) return(numeric())
  exps <- seq.int(floor(log10(lower)), ceiling(log10(upper)), by = 1L)
  ticks <- 10^exps
  ticks[ticks >= lower & ticks <= upper]
}

signed_log_axis_ticks <- function(raw_lim, floor_value = 1e-8) {
  raw_lim <- as_num(raw_lim)
  max_abs <- max(abs(raw_lim), na.rm = TRUE)
  if (!is.finite(max_abs) || max_abs <= 0) return(0)
  min_abs <- max(floor_value, 1e-12)
  exps <- seq.int(floor(log10(min_abs)), ceiling(log10(max_abs)), by = 1L)
  pos_ticks <- 10^exps
  pos_ticks <- pos_ticks[pos_ticks >= min_abs & pos_ticks <= max_abs]
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

draw_error_bars <- function(x, y, sem, col, y_scale = "linear", floor_value = 1e-8) {
  keep <- is.finite(x) & is.finite(y) & is.finite(sem) & sem > 0
  if (any(keep)) {
    lo <- y[keep] - sem[keep]
    hi <- y[keep] + sem[keep]
    if (identical(y_scale, "log10")) {
      lo <- pmax(lo, floor_value)
      hi <- pmax(hi, floor_value)
    }
    graphics::arrows(
      x[keep],
      transform_y_values(lo, y_scale, floor_value),
      x[keep],
      transform_y_values(hi, y_scale, floor_value),
      angle = 90, code = 3, length = 0.025, col = col, lwd = 0.7
    )
  }
}

draw_obsstd_label <- function(obsstd) {
  plot.new()
  text(0.5, 0.5, sprintf("obs std\n%s", num_label(obsstd)), srt = 90, cex = 1)
}

draw_row_label <- function(label, value) {
  plot.new()
  text(0.5, 0.5, sprintf("%s\n%s", label, num_label(value)), srt = 90, cex = 1)
}

draw_header <- function(title, params, colors) {
  plot.new()
  text(0.5, 0.75, title, font = 2, cex = 1)
  if (length(params) == 0L) return(invisible(NULL))
  x0 <- seq(0.12, 0.88, length.out = length(params))
  points(x0, rep(0.34, length(params)), pch = 16, col = colors[as.character(params)], cex = 0.9)
  text(x0, rep(0.14, length(params)), labels = vapply(params, num_label, character(1)), cex = 0.8)
}

plot_kernel_panel <- function(dat, family, ylab, colors, params, y_lim, y_scale = "linear", y_floor = 1e-8) {
  panel_data <- dat[dat$family == family, , drop = FALSE]
  plot_y_lim <- transform_y_values(y_lim, y_scale, y_floor)
  graphics::plot(NA, NA, xlim = c(1, max_observations), ylim = plot_y_lim, xlab = "Observation\nposition",
                 ylab = ylab, xaxt = "n", yaxt = "n", las = 1)
  axis(1, at = seq_len(max_observations))
  draw_scaled_y_axis(y_lim, y_scale, y_floor)
  if (!identical(y_scale, "log10")) {
    abline(h = transform_y_values(0, y_scale, y_floor), col = "grey85", lwd = 0.7)
  }
  if (nrow(panel_data) == 0L) {
    text(mean(par("usr")[1:2]), mean(par("usr")[3:4]), "No data", cex = 0.8, col = "grey40")
    return(invisible(NULL))
  }
  checkpoint_values <- sort(unique(as.character(panel_data$checkpoint)))
  ltys <- seq_along(checkpoint_values)
  names(ltys) <- checkpoint_values
  for (param in params) {
    param_data <- panel_data[parameter_equal(panel_data$parameter_value, param), , drop = FALSE]
    if (nrow(param_data) == 0L) next
    for (checkpoint in checkpoint_values) {
      line_data <- param_data[as.character(param_data$checkpoint) == checkpoint, , drop = FALSE]
      if (nrow(line_data) == 0L) next
      line_data <- line_data[order(line_data$timestep), , drop = FALSE]
      col <- colors[[as.character(param)]]
      lty <- ltys[[checkpoint]]
      plot_y <- transform_y_values(line_data$mean_weight, y_scale, y_floor)
      lines(line_data$timestep, plot_y, col = col, lwd = 1.1, lty = lty)
      points(line_data$timestep, plot_y, col = col, pch = if (family == "beta") 16 else 17, cex = 0.7)
      draw_error_bars(line_data$timestep, line_data$mean_weight, line_data$se_across_runs, col, y_scale, y_floor)
    }
  }
}

panel_y_limits <- function(dat, y_scale = "linear") {
  y_values <- c(dat$mean_weight, dat$mean_weight - dat$se_across_runs, dat$mean_weight + dat$se_across_runs)
  if (identical(y_scale, "log10")) {
    positive_log_axis_limit(y_values)
  } else if (identical(y_scale, "signed_log10")) {
    signed_log_axis_limit(y_values)
  } else {
    safe_range(y_values)
  }
}

save_kernel_plot_one <- function(
  dat,
  slug,
  ylab,
  y_scale = "linear",
  total_obs_label = NULL,
  per_panel_ylim = FALSE,
  row_col = "observation_noise_std",
  row_label = "obs std"
) {
  if (nrow(dat) == 0L) {
    warning(sprintf("No data for plot %s; skipping.", slug))
    return(invisible(NULL))
  }
  if (!row_col %in% names(dat)) {
    warning(sprintf("Plot %s is missing row column %s; skipping.", slug, row_col))
    return(invisible(NULL))
  }
  row_levels <- sort(unique(as_num(dat[[row_col]])))
  row_levels <- row_levels[is.finite(row_levels)]
  if (length(row_levels) == 0L) {
    warning(sprintf("Plot %s has no finite row levels for %s; skipping.", slug, row_col))
    return(invisible(NULL))
  }
  beta_params <- sort(unique(as_num(memory_lambda_values)))
  opp_params <- sort(unique(as_num(opportunity_values)))
  beta_colors <- family_color_values("beta", beta_params)
  opp_colors <- family_color_values("opportunity", opp_params)
  y_values <- c(dat$mean_weight, dat$mean_weight - dat$se_across_runs, dat$mean_weight + dat$se_across_runs)
  y_floor <- if (identical(y_scale, "linear")) 1e-8 else positive_log_floor(abs(y_values))
  y_lim <- panel_y_limits(dat, y_scale)
  n_rows <- length(row_levels)
  layout_matrix <- matrix(0L, nrow = n_rows + 1L, ncol = 3L)
  layout_matrix[1L, ] <- c(0L, 1L, 2L)
  next_id <- 3L
  for (i in seq_len(n_rows)) {
    layout_matrix[i + 1L, ] <- c(next_id, next_id + 1L, next_id + 2L)
    next_id <- next_id + 3L
  }
  panel_cell_width_in <- target_panel_side_in + panel_margin_in[["left"]] + panel_margin_in[["right"]]
  panel_cell_height_in <- target_panel_side_in + panel_margin_in[["bottom"]] + panel_margin_in[["top"]]
  file_slug <- if (is.null(total_obs_label)) slug else sprintf("%s_totalobs_%s", slug, value_token(total_obs_label))
  output_file <- file.path(output_dir, sprintf("evidence_accumulation_integration_weights_%s.png", file_slug))
  grDevices::png(
    output_file,
    width = label_col_width_in + 2 * panel_cell_width_in,
    height = header_row_height_in + n_rows * panel_cell_height_in,
    units = "in",
    res = 300,
    pointsize = 7
  )
  layout(
    layout_matrix,
    widths = c(label_col_width_in / panel_cell_width_in, 1, 1),
    heights = c(header_row_height_in / panel_cell_height_in, rep(1, n_rows))
  )
  old_par <- par(no.readonly = TRUE)
  par(cex = 1, cex.axis = 1, cex.lab = 1, cex.main = 1, oma = c(0, 0, 0, 0), xpd = FALSE)
  par(mai = c(0.02, 0.02, 0.02, 0.04))
  draw_header(sprintf("Varying memory lambda\nopp = %s", num_label(fixed_opp)), beta_params, beta_colors)
  par(mai = c(0.02, 0.02, 0.02, 0.04))
  draw_header(sprintf("Varying opportunity\nmemory lambda = %s", num_label(fixed_memory_lambda)), opp_params, opp_colors)
  for (i in seq_along(row_levels)) {
    row_value <- row_levels[[i]]
    row_data <- dat[parameter_equal(dat[[row_col]], row_value), , drop = FALSE]
    par(mai = label_margin_in)
    draw_row_label(row_label, row_value)
    par(mai = panel_margin_in)
    beta_data <- row_data[row_data$family == "beta", , drop = FALSE]
    beta_y_lim <- if (per_panel_ylim && nrow(beta_data) > 0L) panel_y_limits(beta_data, y_scale) else y_lim
    plot_kernel_panel(row_data, "beta", ylab, beta_colors, beta_params, beta_y_lim, y_scale, y_floor)
    par(mai = panel_margin_in)
    opp_data <- row_data[row_data$family == "opportunity", , drop = FALSE]
    opp_y_lim <- if (per_panel_ylim && nrow(opp_data) > 0L) panel_y_limits(opp_data, y_scale) else y_lim
    plot_kernel_panel(row_data, "opportunity", ylab, opp_colors, opp_params, opp_y_lim, y_scale, y_floor)
  }
  invisible(try(par(old_par), silent = TRUE))
  grDevices::dev.off()
  message(sprintf("Saved %s", output_file))
  invisible(output_file)
}

save_kernel_plot <- function(
  dat,
  slug,
  ylab,
  y_scale = "linear",
  per_panel_ylim = FALSE,
  row_col = "observation_noise_std",
  row_label = "obs std"
) {
  if (nrow(dat) == 0L) {
    warning(sprintf("No data for plot %s; skipping.", slug))
    return(invisible(NULL))
  }
  if (!"num_observations" %in% names(dat)) {
    return(save_kernel_plot_one(dat, slug, ylab, y_scale, per_panel_ylim = per_panel_ylim, row_col = row_col, row_label = row_label))
  }
  levels <- sort(unique(as_num(dat$num_observations)))
  levels <- levels[is.finite(levels)]
  if (length(levels) == 0L) {
    return(save_kernel_plot_one(dat, slug, ylab, y_scale, per_panel_ylim = per_panel_ylim, row_col = row_col, row_label = row_label))
  }
  outputs <- list()
  for (n_obs in levels) {
    subset <- dat[parameter_equal(dat$num_observations, n_obs), , drop = FALSE]
    if (nrow(subset) == 0L) next
    outputs[[length(outputs) + 1L]] <- save_kernel_plot_one(
      subset,
      slug,
      ylab,
      y_scale,
      total_obs_label = n_obs,
      per_panel_ylim = per_panel_ylim,
      row_col = row_col,
      row_label = row_label
    )
  }
  invisible(outputs)
}

kernel_metric_label <- function(metric_name) {
  switch(
    metric_name,
    raw = "Raw logistic\ncoefficient",
    standardized = "Standardized\ncoefficient",
    normalized = "Normalized\nweight",
    "Integration\nweight"
  )
}

save_kernel_metric_set <- function(
  summary_dat,
  metrics = names(metric_cols),
  per_panel_raw_ylim = TRUE,
  row_col = "observation_noise_std",
  row_label = "obs std"
) {
  for (metric_name in metrics) {
    metric_data <- if (nrow(summary_dat) > 0L && "metric" %in% names(summary_dat)) {
      summary_dat[summary_dat$metric == metric_name, , drop = FALSE]
    } else {
      data.frame()
    }
    save_kernel_plot(
      metric_data,
      metric_name,
      kernel_metric_label(metric_name),
      per_panel_ylim = isTRUE(per_panel_raw_ylim) && identical(metric_name, "raw"),
      row_col = row_col,
      row_label = row_label
    )
  }
}

save_simple_kernel_plots <- function() {
  if (!simple_kernel_mode) {
    return(invisible(NULL))
  }
  if (nrow(summary_weights) == 0L || !"coherence_magnitude" %in% names(summary_weights)) {
    warning("Simple integration-weight plots require non-pooled coherence_magnitude in the summary; skipping.")
    return(invisible(NULL))
  }
  simple_dir_name <- if (!is.null(simple_output_subdir_arg) && nzchar(trim_string(simple_output_subdir_arg))) {
    trim_string(simple_output_subdir_arg)
  } else if (length(observer_modes) == 1L && isTRUE(observer_modes[[1L]])) {
    "observer_only_simple"
  } else if (length(observer_modes) == 1L && identical(observer_modes[[1L]], FALSE)) {
    "no_observer_simple"
  } else {
    "observer_mixed_simple"
  }
  simple_values <- simple_coherence_values
  if (length(simple_values) == 0L) {
    simple_values <- sort(unique(as_num(summary_weights$coherence_magnitude)))
    simple_values <- simple_values[is.finite(simple_values)]
  }
  keep_simple <- parameter_equal(summary_weights$observation_noise_std, simple_fixed_obsstd)
  if (length(simple_values) > 0L) {
    keep_coh <- rep(FALSE, nrow(summary_weights))
    for (coh in simple_values) {
      keep_coh <- keep_coh | parameter_equal(summary_weights$coherence_magnitude, coh)
    }
    keep_simple <- keep_simple & keep_coh
  }
  simple_dat <- summary_weights[keep_simple, , drop = FALSE]
  if (nrow(simple_dat) == 0L) {
    warning(sprintf(
      "No simple integration-weight summary rows for obsstd=%s, coherence=%s; skipping.",
      num_label(simple_fixed_obsstd),
      paste(vapply(simple_values, num_label, character(1)), collapse = ",")
    ))
    return(invisible(NULL))
  }
  old_output_dir <- output_dir
  on.exit({
    output_dir <<- old_output_dir
  }, add = TRUE)
  output_dir <<- file.path(output_base_dir, simple_dir_name)
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  message(sprintf(
    "Saving simple integration-weight plots to %s for obsstd=%s; coherence rows=%s.",
    output_dir,
    num_label(simple_fixed_obsstd),
    paste(vapply(sort(unique(as_num(simple_dat$coherence_magnitude))), num_label, character(1)), collapse = ",")
  ))
  save_kernel_metric_set(
    simple_dat,
    metrics = c("raw", "standardized", "normalized"),
    per_panel_raw_ylim = TRUE,
    row_col = "coherence_magnitude",
    row_label = "coherence"
  )
  invisible(NULL)
}

if (simple_kernel_mode) {
  save_simple_kernel_plots()
} else {
  save_kernel_metric_set(summary_weights, metrics = names(metric_cols), per_panel_raw_ylim = TRUE)

  if (nrow(logit_summary) > 0L) {
    save_kernel_plot(logit_summary, "choice_aligned_choice_logit", "Choice-aligned\nA/B logit")
  }

  timecourse_plot_labels <- c(
    choice_aligned_cumulative_evidence = "Choice-aligned\ncumulative evidence",
    delta_kl = "Delta KL\npaid",
    abs_delta_z_mu = "|delta prior-norm\nz_mu|",
    signed_delta_z_sigma = "Signed delta\nprior-norm z_sigma",
    abs_delta_action_aligned_action_logit = "|delta action-aligned\nA/B logit|"
  )
  if (nrow(timecourse_summary) > 0L && "metric" %in% names(timecourse_summary)) {
    for (metric_name in names(timecourse_plot_labels)) {
      metric_data <- timecourse_summary[timecourse_summary$metric == metric_name, , drop = FALSE]
      y_scale <- switch(
        metric_name,
        delta_kl = "log10",
        abs_delta_z_mu = "log10",
        signed_delta_z_sigma = "signed_log10",
        "linear"
      )
      save_kernel_plot(metric_data, metric_name, timecourse_plot_labels[[metric_name]], y_scale = y_scale)
    }
  }
}

message(sprintf("Saved integration weights CSV to: %s", weights_path))
message(sprintf("Saved integration weight summary CSV to: %s", summary_path))
if (nrow(timecourse_summary) > 0L) {
  message(sprintf("Saved duration-controlled timecourse summary CSV to: %s", timecourse_summary_path))
}
if (nrow(contrasts) > 0L) message(sprintf("Saved bump-contrast CSV to: %s", contrast_path))
if (all(as.logical(trial_data$choice_at_end_only))) {
  message("Observer-only outputs are fixed duration; opportunity cost is metadata for these forced-duration fits.")
} else {
  message("Self-timed outputs are grouped by realized total observations before stopping.")
}
