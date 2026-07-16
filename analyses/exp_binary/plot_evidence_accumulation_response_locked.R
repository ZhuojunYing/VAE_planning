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
    "  Rscript analyses/exp_binary/plot_evidence_accumulation_response_locked.R <task> [options]\n\n",
    "Creates one response-locked evidence-accumulation PNG per analysis from\n",
    "trial-level outputs written by model_jax/evidence_accumulation.py. Each PNG\n",
    "uses rows for observation-noise std and columns for beta-vary / opportunity-vary.\n\n",
    "Options:\n",
    "  --preset-file PATH          Preset CSV path. Default: analyses/exp_binary/evidence_accumulation_plot_presets.csv.\n",
    "  --input-dir DIR             Simulation CSV directory. Default comes from the preset.\n",
    "  --output-file PATH          Output PNG stem/path. Plot type suffixes are added.\n",
    "  --output-root DIR           Output root if --output-file is omitted. Default comes from the preset or results.\n",
    "  --vary-beta-values LIST     Beta values for the left column. Aliases: --beta-values, --betas.\n",
    "  --vary-opportunity-values LIST\n",
    "                              Opportunity costs for the right column. Aliases: --opportunity-values, --opps.\n",
    "  --fixed-beta VALUE          Beta held fixed for the opportunity-vary column.\n",
    "  --fixed-opp VALUE           Opportunity cost held fixed for the beta-vary column.\n",
    "  --fixed-coherence VALUE     Coherence magnitude to hold fixed. If omitted, uses the most common nonzero value.\n",
    "  --observation-noise-std LIST\n",
    "                              Observation noise std value(s) to plot; multiple values add rows.\n",
    "                              Aliases: --obsstd, --sigma, --fixed-observation-noise-std.\n",
    "  --lambda VALUE              Filter lambda value. Default comes from preset when available.\n",
    "  --alpha VALUE               Filter alpha value. Default comes from preset when available.\n",
    "  --seeds LIST                Filter seeds. Default comes from preset when available.\n",
    "  --rnn-units VALUE           Filter RNN units. Default comes from preset when available.\n",
    "  --latent-dim VALUE          Filter latent dimension. Default comes from preset when available.\n",
    "  --max-observations VALUE    Filter max observations. Alias: --maxobs. Default comes from preset when available.\n",
    "  --input-type VALUE          Filter trailing input type. Default: evidence.\n",
    "  --pay-kl-on-stop            Use CSVs with the _stop_paid filename suffix. Default comes from preset, now true.\n",
    "  --no-pay-kl-on-stop         Use legacy CSVs without the _stop_paid filename suffix.\n",
    "  --max-steps-before-stop N   Keep relative timesteps -N,...,0. Default: 10.\n",
    "  --include-forced-stops      Include trials that stop at the forced max-observation decision. Default excludes them.\n",
    "  --min-samples N             Drop summarized points with fewer than N contributing trials. Default: 10.\n",
    "  --help                      Show this message.\n\n",
    "Example:\n",
    "  Rscript analyses/exp_binary/plot_evidence_accumulation_response_locked.R evidence \\\n",
    "    --vary-beta-values \"10,20,80\" \\\n",
    "    --vary-opportunity-values \"0.06,0.2,0.4\" \\\n",
    "    --fixed-coherence 0.2 --observation-noise-std \"0.1,0.5,1\"\n",
    sep = ""
  )
}

if (length(args) == 0L || any(args %in% c("--help", "-h"))) {
  usage()
  quit(save = "no", status = if (length(args) == 0L) 1L else 0L)
}

trim_string <- function(value) trimws(as.character(value))
as_num <- function(value) suppressWarnings(as.numeric(as.character(value)))

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

extract_boolean_flag <- function(args, option_names, default = FALSE) {
  value <- default
  keep <- rep(TRUE, length(args))
  truthy <- c("1", "true", "t", "yes", "y", "on")
  falsey <- c("0", "false", "f", "no", "n", "off")
  for (i in seq_along(args)) {
    arg <- args[[i]]
    for (option_name in option_names) {
      if (identical(arg, option_name)) {
        value <- TRUE
        keep[[i]] <- FALSE
      } else if (startsWith(arg, paste0(option_name, "="))) {
        raw <- tolower(trim_string(sub(paste0("^", option_name, "="), "", arg)))
        if (raw %in% truthy) {
          value <- TRUE
        } else if (raw %in% falsey) {
          value <- FALSE
        } else {
          stop(sprintf("%s expects true/false when using --flag=value syntax.", option_name))
        }
        keep[[i]] <- FALSE
      }
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

num_label <- function(value) {
  value_num <- suppressWarnings(as.numeric(value))
  if (!is.finite(value_num)) {
    return(as.character(value))
  }
  label <- format(value_num, scientific = FALSE, trim = TRUE, digits = 8)
  if (grepl("\\.", label)) {
    label <- sub("0+$", "", label)
    label <- sub("\\.$", "", label)
  }
  if (!nzchar(label)) "0" else label
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

preset_file_option <- extract_named_option(args, c("--preset-file"), default = file.path(script_dir, "evidence_accumulation_plot_presets.csv"))
args <- preset_file_option$args
input_dir_option <- extract_named_option(args, c("--input-dir"), default = NULL)
args <- input_dir_option$args
output_file_option <- extract_named_option(args, c("--output-file", "--out"), default = NULL)
args <- output_file_option$args
output_root_option <- extract_named_option(args, c("--output-root", "--results-dir"), default = NULL)
args <- output_root_option$args
beta_values_option <- extract_named_option(args, c("--vary-beta-values", "--beta-values", "--betas", "--vary-betas"), default = NULL)
args <- beta_values_option$args
opportunity_values_option <- extract_named_option(
  args,
  c("--vary-opportunity-values", "--vary-opportunities", "--vary-opps", "--opportunity-values", "--opportunities", "--opps", "--opportunity-costs"),
  default = NULL
)
args <- opportunity_values_option$args
fixed_beta_option <- extract_named_option(args, c("--fixed-beta"), default = NULL)
args <- fixed_beta_option$args
fixed_opp_option <- extract_named_option(args, c("--fixed-opp", "--fixed-opportunity", "--fixed-opportunity-cost"), default = NULL)
args <- fixed_opp_option$args
fixed_coherence_option <- extract_named_option(args, c("--fixed-coherence", "--coherence"), default = NULL)
args <- fixed_coherence_option$args
fixed_obsstd_option <- extract_named_option(args, c("--observation-noise-std", "--obsstd", "--sigma", "--fixed-observation-noise-std", "--fixed-obsstd", "--fixed-sigma"), default = NULL)
args <- fixed_obsstd_option$args
lambda_option <- extract_named_option(args, c("--lambda", "--lambda-value"), default = NULL)
args <- lambda_option$args
alpha_option <- extract_named_option(args, c("--alpha"), default = NULL)
args <- alpha_option$args
seeds_option <- extract_named_option(args, c("--seeds"), default = NULL)
args <- seeds_option$args
rnn_option <- extract_named_option(args, c("--rnn-units", "--rnn-dims", "--rnn-dim"), default = NULL)
args <- rnn_option$args
latent_option <- extract_named_option(args, c("--latent-dim", "--latent-dims"), default = NULL)
args <- latent_option$args
maxobs_option <- extract_named_option(args, c("--max-observations", "--max-observations-before-stop", "--maxobs"), default = NULL)
args <- maxobs_option$args
input_type_option <- extract_named_option(args, c("--input-type"), default = NULL)
args <- input_type_option$args
pay_kl_on_stop_option <- extract_boolean_option(
  args,
  c("--pay-kl-on-stop", "--stop-paid"),
  c("--no-pay-kl-on-stop", "--no-stop-paid", "--legacy-no-stop-paid"),
  default = NULL
)
args <- pay_kl_on_stop_option$args
max_steps_option <- extract_named_option(args, c("--max-steps-before-stop", "--max-relative-steps"), default = "10")
args <- max_steps_option$args
min_samples_option <- extract_named_option(args, c("--min-samples", "--min-n"), default = "10")
args <- min_samples_option$args
include_forced_option <- extract_boolean_flag(args, c("--include-forced-stops", "--include-forced-terminal"), default = FALSE)
args <- include_forced_option$args

if (length(args) != 1L) {
  usage()
  stop("Expected exactly one positional argument: <task>, usually evidence.")
}

task_arg <- trim_string(args[[1L]])
max_steps_before_stop <- as.integer(round(as.numeric(max_steps_option$value)))
minimum_samples <- as.integer(round(as.numeric(min_samples_option$value)))
if (!is.finite(max_steps_before_stop) || max_steps_before_stop < 0L) {
  stop("--max-steps-before-stop must be a nonnegative integer.")
}
if (!is.finite(minimum_samples) || minimum_samples < 0L) {
  stop("--min-samples must be a nonnegative integer.")
}
include_forced_stops <- isTRUE(include_forced_option$value)

load_preset_rows <- function(preset_file, task) {
  if (!file.exists(preset_file)) {
    return(NULL)
  }
  presets <- utils::read.csv(preset_file, stringsAsFactors = FALSE, check.names = FALSE)
  if (!all(c("task", "vary") %in% names(presets))) {
    return(NULL)
  }
  rows <- presets[trimws(presets$task) == task, , drop = FALSE]
  beta_row <- rows[trimws(rows$vary) == "beta", , drop = FALSE]
  opp_row <- rows[trimws(rows$vary) == "opportunity", , drop = FALSE]
  if (nrow(beta_row) == 0L || nrow(opp_row) == 0L) {
    return(NULL)
  }
  list(beta = beta_row[1L, , drop = FALSE], opportunity = opp_row[1L, , drop = FALSE])
}

preset_value <- function(rows, which, column, default = NULL) {
  if (is.null(rows) || !column %in% names(rows[[which]])) {
    return(default)
  }
  value <- rows[[which]][[column]][[1L]]
  if (is.na(value) || !nzchar(trim_string(value))) {
    return(default)
  }
  trim_string(value)
}

preset_rows <- load_preset_rows(preset_file_option$value, task_arg)
if (!is.null(preset_rows)) {
  message(sprintf("Using evidence accumulation preset: task=%s from %s", task_arg, preset_file_option$value))
}

if (is.null(input_dir_option$value)) input_dir_option$value <- preset_value(preset_rows, "beta", "input_dir", "outputs/jax_simulations_evi")
if (is.null(output_root_option$value)) output_root_option$value <- preset_value(preset_rows, "beta", "results_dir", "results")
if (is.null(input_type_option$value)) input_type_option$value <- preset_value(preset_rows, "beta", "input_type", "evidence")
if (is.null(beta_values_option$value)) beta_values_option$value <- preset_value(preset_rows, "beta", "beta_arg", NULL)
if (is.null(opportunity_values_option$value)) opportunity_values_option$value <- preset_value(preset_rows, "opportunity", "opportunity_arg", NULL)
if (is.null(lambda_option$value)) lambda_option$value <- preset_value(preset_rows, "beta", "lambda_arg", NULL)
if (is.null(alpha_option$value)) alpha_option$value <- preset_value(preset_rows, "beta", "alpha_arg", NULL)
if (is.null(seeds_option$value)) seeds_option$value <- preset_value(preset_rows, "beta", "seed_arg", NULL)
if (is.null(rnn_option$value)) rnn_option$value <- preset_value(preset_rows, "beta", "rnn_units_arg", NULL)
if (is.null(latent_option$value)) latent_option$value <- preset_value(preset_rows, "beta", "latent_dim_arg", NULL)
if (is.null(maxobs_option$value)) maxobs_option$value <- preset_value(preset_rows, "beta", "max_observations_arg", NULL)
if (is.null(fixed_obsstd_option$value)) fixed_obsstd_option$value <- preset_value(preset_rows, "beta", "observation_noise_std_arg", NULL)
if (is.null(pay_kl_on_stop_option$value)) {
  pay_kl_on_stop_option$value <- parse_bool_value(
    preset_value(preset_rows, "beta", "pay_kl_on_stop_arg", "true"),
    default = TRUE,
    label = "pay_kl_on_stop_arg"
  )
}

input_dir <- input_dir_option$value
output_root <- output_root_option$value
input_type <- trim_string(input_type_option$value)
pay_kl_on_stop_mode <- isTRUE(pay_kl_on_stop_option$value)

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

parse_evidence_filename_index <- function(input_dir) {
  files <- list.files(input_dir, pattern = "_evidence\\.csv$", recursive = TRUE, full.names = TRUE)
  files <- files[!grepl("_evidence_summary\\.csv$", basename(files))]
  if (length(files) == 0L) {
    return(data.frame())
  }
  basenames <- basename(files)
  pattern <- paste0(
    "^evidence_lambda_([^_]+)_alpha_([^_]+)_beta_([^_]+)_opportunity_([^_]+)",
    "_expansion_([^_]+)_variant_([^_]+)_seed_([0-9]+)_coh_n([0-9]+)",
    "_min_([^_]+)_max_([^_]+)_obsstd_([^_]+)_maxobs_([0-9]+)",
    "_rnn_([^_]+)_latent_([^_]+)(_stop_paid)?_(.+)\\.csv$"
  )
  matches <- regexec(pattern, basenames, perl = TRUE)
  parts <- regmatches(basenames, matches)
  keep <- lengths(parts) > 0L
  if (!any(keep)) {
    return(data.frame())
  }
  parts <- parts[keep]
  files <- files[keep]
  part_at <- function(index) {
    vapply(parts, function(x) {
      if (length(x) >= index && !is.na(x[[index]])) x[[index]] else ""
    }, character(1))
  }
  data.frame(
    file = files,
    lambda = as_num(part_at(2L)),
    alpha = as_num(part_at(3L)),
    beta = as_num(part_at(4L)),
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
    pay_kl_on_stop = nzchar(part_at(16L)),
    input_type = part_at(17L),
    stringsAsFactors = FALSE
  )
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

most_common_numeric <- function(values, prefer_nonzero = FALSE, label = "value") {
  values <- as_num(values)
  values <- values[is.finite(values)]
  if (prefer_nonzero) {
    nonzero <- values[abs(values) > 1e-12]
    if (length(nonzero) > 0L) {
      values <- nonzero
    }
  }
  if (length(values) == 0L) {
    stop(sprintf("No available %s values.", label))
  }
  formatted <- format(values, digits = 16, scientific = FALSE, trim = TRUE)
  tab <- sort(table(formatted), decreasing = TRUE)
  as.numeric(names(tab)[[1L]])
}

select_or_most_common <- function(values, requested, label, prefer_nonzero = FALSE) {
  values <- as_num(values)
  values <- values[is.finite(values)]
  if (!is.null(requested) && nzchar(trim_string(requested))) {
    value <- as_num(parse_csv_values(requested)[[1L]])
    if (!any(parameter_equal(values, value))) {
      stop(sprintf("Requested %s=%s not found. Available: %s", label, num_label(value), values_label(values)))
    }
    message(sprintf("Selected requested %s=%s.", label, num_label(value)))
    return(value)
  }
  value <- most_common_numeric(values, prefer_nonzero = prefer_nonzero, label = label)
  message(sprintf("No %s supplied; selected most common%s %s=%s.",
    label,
    if (prefer_nonzero) " nonzero" else "",
    label,
    num_label(value)
  ))
  value
}

values_from_option_or_available <- function(requested, available, label) {
  available <- sort(unique(as_num(available)))
  available <- available[is.finite(available)]
  if (!is.null(requested) && nzchar(trim_string(requested))) {
    values <- as_num(parse_csv_values(requested))
    values <- values[is.finite(values)]
    missing <- values[!vapply(values, function(v) any(parameter_equal(available, v)), logical(1))]
    if (length(missing) > 0L) {
      stop(sprintf("Requested %s value(s) not found: %s. Available: %s", label, values_label(missing), values_label(available)))
    }
    return(sort(unique(values)))
  }
  available
}

if (!dir.exists(input_dir)) {
  stop(sprintf("Simulation input directory not found: %s", input_dir))
}

file_index <- parse_evidence_filename_index(input_dir)
if (nrow(file_index) == 0L) {
  stop(sprintf("No evidence simulation CSVs were found under %s.", input_dir))
}
file_index <- file_index[file_index$input_type == input_type, , drop = FALSE]
file_index <- filter_numeric_option(file_index, "lambda", lambda_option$value, "--lambda")
file_index <- filter_numeric_option(file_index, "alpha", alpha_option$value, "--alpha")
file_index <- filter_numeric_option(file_index, "seed", seeds_option$value, "--seeds")
file_index <- filter_numeric_option(file_index, "rnn_units", rnn_option$value, "--rnn-units")
file_index <- filter_numeric_option(file_index, "latent_dim", latent_option$value, "--latent-dim")
file_index <- filter_numeric_option(file_index, "max_observations", maxobs_option$value, "--max-observations")
file_index <- file_index[file_index$pay_kl_on_stop == pay_kl_on_stop_mode, , drop = FALSE]
if (nrow(file_index) == 0L) {
  stop(sprintf(
    "No evidence simulation CSVs remain after metadata and stop-paid filters. Requested pay_kl_on_stop=%s.",
    if (pay_kl_on_stop_mode) "true" else "false"
  ))
}

if (!is.null(fixed_obsstd_option$value) && nzchar(trim_string(fixed_obsstd_option$value))) {
  selected_obsstd_values <- values_from_option_or_available(
    fixed_obsstd_option$value,
    file_index$observation_noise_std,
    "observation_noise_std"
  )
  keep_obsstd <- rep(FALSE, nrow(file_index))
  for (obsstd in selected_obsstd_values) {
    keep_obsstd <- keep_obsstd | parameter_equal(file_index$observation_noise_std, obsstd)
  }
  file_index <- file_index[keep_obsstd, , drop = FALSE]
  if (nrow(file_index) == 0L) {
    stop("No files remain after observation-noise filter.")
  }
} else {
  selected_obsstd <- most_common_numeric(file_index$observation_noise_std, label = "observation_noise_std")
  selected_obsstd_values <- selected_obsstd
  file_index <- file_index[parameter_equal(file_index$observation_noise_std, selected_obsstd), , drop = FALSE]
  message(sprintf("No observation noise was supplied; selected most common obsstd=%s.", num_label(selected_obsstd)))
}
selected_obsstd_values <- sort(unique(as_num(selected_obsstd_values)))
selected_obsstd_values <- selected_obsstd_values[is.finite(selected_obsstd_values)]
if (length(selected_obsstd_values) == 0L) {
  stop("No valid observation-noise std values were selected.")
}

fixed_opp <- select_or_most_common(file_index$opportunity, fixed_opp_option$value, "opportunity")
fixed_beta <- select_or_most_common(file_index$beta, fixed_beta_option$value, "beta")

beta_values <- values_from_option_or_available(
  beta_values_option$value,
  file_index$beta[parameter_equal(file_index$opportunity, fixed_opp)],
  "beta"
)
opportunity_values <- values_from_option_or_available(
  opportunity_values_option$value,
  file_index$opportunity[parameter_equal(file_index$beta, fixed_beta)],
  "opportunity"
)
if (length(beta_values) < 2L) {
  stop(sprintf("Need at least two beta values for the beta-vary column. Found: %s", values_label(beta_values)))
}
if (length(opportunity_values) < 2L) {
  stop(sprintf("Need at least two opportunity values for the opportunity-vary column. Found: %s", values_label(opportunity_values)))
}

manifest_beta <- file_index[parameter_equal(file_index$opportunity, fixed_opp), , drop = FALSE]
manifest_beta <- manifest_beta[vapply(manifest_beta$beta, function(v) any(parameter_equal(beta_values, v)), logical(1)), , drop = FALSE]
manifest_opp <- file_index[parameter_equal(file_index$beta, fixed_beta), , drop = FALSE]
manifest_opp <- manifest_opp[vapply(manifest_opp$opportunity, function(v) any(parameter_equal(opportunity_values, v)), logical(1)), , drop = FALSE]
manifest_beta$comparison_type <- "vary_beta"
manifest_beta$parameter_value <- manifest_beta$beta
manifest_opp$comparison_type <- "vary_opportunity"
manifest_opp$parameter_value <- manifest_opp$opportunity
manifest <- rbind(manifest_beta, manifest_opp)
if (nrow(manifest) == 0L) {
  stop("No matching files found for the selected beta/opportunity comparison.")
}

message(sprintf("Loaded manifest with %d input file(s).", nrow(manifest)))
message(sprintf("Available beta values after filters: %s", values_label(file_index$beta)))
message(sprintf("Available opportunity-cost values after filters: %s", values_label(file_index$opportunity)))
message(sprintf("Available observation-noise values after filters: %s", values_label(file_index$observation_noise_std)))
message(sprintf("Observation noise std values included in figures: %s", values_label(selected_obsstd_values)))
message(sprintf("Stop-paid KL file mode: %s", if (pay_kl_on_stop_mode) "using _stop_paid CSVs" else "using legacy/no _stop_paid CSVs"))
message(sprintf("Independent run/file count in manifest: %d", length(unique(basename(manifest$file)))))
message(sprintf("Selected fixed beta: %s", num_label(fixed_beta)))
message(sprintf("Selected fixed opportunity cost: %s", num_label(fixed_opp)))

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

load_evidence_accumulation_results <- function(path, meta) {
  cols <- read_csv_names(path)
  required_base <- c(
    "graph", "coherence", "signed_coherence", "num_observations", "stopping_time",
    "terminal_action", "choose_right", "choose_correct"
  )
  timestep_cols <- grep(
    "^(evidence_sample|cumulative_evidence|policy_continue|policy_choose_a|policy_choose_b|action|stop)_t[0-9]+$",
    cols,
    value = TRUE
  )
  selected <- unique(intersect(c(required_base, timestep_cols), cols))
  dat <- read_csv_fast(path, select = selected)
  missing <- setdiff(c("coherence", "num_observations", "terminal_action"), names(dat))
  if (length(missing) > 0L) {
    stop(sprintf("Missing required column(s) in %s: %s", path, paste(missing, collapse = ", ")))
  }
  for (name in c("policy_continue_t1", "policy_choose_a_t1", "policy_choose_b_t1")) {
    if (!name %in% names(dat)) {
      stop(sprintf("Missing policy probability columns in %s; expected %s and corresponding t* columns.", path, name))
    }
  }
  if (!any(grepl("^evidence_sample_t[0-9]+$", names(dat)))) {
    stop(sprintf("Missing evidence_sample_t* trajectory columns in %s.", path))
  }
  if (!any(grepl("^cumulative_evidence_t[0-9]+$", names(dat)))) {
    stop(sprintf("Missing cumulative_evidence_t* trajectory columns in %s.", path))
  }
  dat$run_id <- basename(path)
  dat$file <- path
  dat$beta <- meta$beta[[1L]]
  dat$opportunity <- meta$opportunity[[1L]]
  dat$seed <- meta$seed[[1L]]
  dat$lambda <- meta$lambda[[1L]]
  dat$alpha <- meta$alpha[[1L]]
  dat$observation_noise_std <- meta$observation_noise_std[[1L]]
  dat$max_observations <- meta$max_observations[[1L]]
  dat$comparison_type <- meta$comparison_type[[1L]]
  dat$parameter_value <- meta$parameter_value[[1L]]
  dat
}

loaded <- vector("list", nrow(manifest))
for (i in seq_len(nrow(manifest))) {
  loaded[[i]] <- load_evidence_accumulation_results(manifest$file[[i]], manifest[i, , drop = FALSE])
}
trial_data <- do.call(rbind, loaded)
trial_data$trial_id <- paste(trial_data$run_id, trial_data$graph, sep = "::")
message(sprintf("Number of input files loaded: %d", nrow(manifest)))
message(sprintf("Total number of trials before coherence filtering: %d", nrow(trial_data)))

message(sprintf("Available coherence values: %s", values_label(trial_data$coherence)))
fixed_coherence <- select_or_most_common(
  trial_data$coherence,
  fixed_coherence_option$value,
  "coherence",
  prefer_nonzero = TRUE
)
trial_data <- trial_data[parameter_equal(trial_data$coherence, fixed_coherence), , drop = FALSE]
if (nrow(trial_data) == 0L) {
  stop("No trials remain after fixed coherence filter.")
}
message(sprintf("Selected fixed coherence: %s", num_label(fixed_coherence)))
message(sprintf("Total number of trials after fixed coherence filtering: %d", nrow(trial_data)))

reshape_trajectory_arrays <- function(dat, max_steps_before_stop) {
  cols <- names(dat)
  evidence_cols <- step_columns(cols, "evidence_sample_t")
  cumulative_cols <- step_columns(cols, "cumulative_evidence_t")
  p_continue_cols <- step_columns(cols, "policy_continue_t")
  p_a_cols <- step_columns(cols, "policy_choose_a_t")
  p_b_cols <- step_columns(cols, "policy_choose_b_t")
  action_cols <- step_columns(cols, "action_t")
  stop_cols <- step_columns(cols, "stop_t")
  step_ids <- sort(unique(as.integer(sub(".*_t", "", c(
    evidence_cols, cumulative_cols, p_continue_cols, p_a_cols, p_b_cols, action_cols, stop_cols
  )))))
  step_ids <- step_ids[is.finite(step_ids)]
  if (length(step_ids) == 0L) {
    stop("Unable to identify timestep columns.")
  }
  pieces <- list()
  discrepancy_count <- 0L
  p_norm_errors <- numeric()
  p_stop_values <- numeric()
  stabilized_count <- 0L
  for (step in step_ids) {
    required <- sprintf(
      c("evidence_sample_t%d", "cumulative_evidence_t%d", "policy_continue_t%d", "policy_choose_a_t%d", "policy_choose_b_t%d"),
      step
    )
    if (!all(required %in% names(dat))) {
      next
    }
    terminal_action <- as.integer(round(as_num(dat$terminal_action)))
    if (any(!terminal_action %in% c(1L, 2L), na.rm = TRUE)) {
      stop("Terminal actions could not be mapped to CHOOSE_A=1 and CHOOSE_B=2.")
    }
    stop_time_saved <- as.integer(round(as_num(dat$num_observations)))
    stop_col <- sprintf("stop_t%d", step)
    action_col <- sprintf("action_t%d", step)
    explicit_stop <- if (stop_col %in% names(dat)) coerce_logical(dat[[stop_col]]) else rep(FALSE, nrow(dat))
    if (step == min(step_ids)) {
      stop_mat <- sapply(step_ids, function(s) {
        col <- sprintf("stop_t%d", s)
        if (col %in% names(dat)) coerce_logical(dat[[col]]) else rep(FALSE, nrow(dat))
      })
      if (!is.matrix(stop_mat)) stop_mat <- matrix(stop_mat, ncol = length(step_ids))
      has_stop <- rowSums(stop_mat, na.rm = TRUE) > 0
      first_stop_idx <- max.col(stop_mat, ties.method = "first")
      first_stop_step <- rep(NA_integer_, nrow(dat))
      first_stop_step[has_stop] <- step_ids[first_stop_idx[has_stop]]
      mismatch <- has_stop & is.finite(stop_time_saved) & first_stop_step != stop_time_saved
      discrepancy_count <- sum(mismatch, na.rm = TRUE)
    }
    valid <- is.finite(stop_time_saved) & step <= stop_time_saved
    valid <- valid & terminal_action %in% c(1L, 2L)
    if (!any(valid)) {
      next
    }
    p_continue <- as_num(dat[[sprintf("policy_continue_t%d", step)]])
    p_a <- as_num(dat[[sprintf("policy_choose_a_t%d", step)]])
    p_b <- as_num(dat[[sprintf("policy_choose_b_t%d", step)]])
    p_stop <- p_a + p_b
    p_norm_errors <- c(p_norm_errors, abs(p_continue[valid] + p_a[valid] + p_b[valid] - 1))
    p_stop_values <- c(p_stop_values, p_stop[valid])
    p_eventual_choice <- ifelse(terminal_action == 1L, p_a, p_b)
    stabilized <- valid & is.finite(p_stop) & p_stop < 1e-12
    stabilized_count <- stabilized_count + sum(stabilized, na.rm = TRUE)
    p_eventual_choice_given_stop <- p_eventual_choice / pmax(p_stop, 1e-12)
    choice_direction <- ifelse(terminal_action == 2L, 1, -1)
    cumulative_evidence <- as_num(dat[[sprintf("cumulative_evidence_t%d", step)]])
    evidence_sample <- as_num(dat[[sprintf("evidence_sample_t%d", step)]])
    relative_timestep <- step - stop_time_saved
    keep <- valid & relative_timestep >= -max_steps_before_stop & relative_timestep <= 0
    if (!any(keep)) {
      next
    }
    forced_terminal <- stop_time_saved >= as.integer(round(as_num(dat$max_observations)))
    pieces[[length(pieces) + 1L]] <- data.frame(
      trial_id = dat$trial_id[keep],
      run_id = dat$run_id[keep],
      beta = dat$beta[keep],
      opportunity_cost = dat$opportunity[keep],
      seed = dat$seed[keep],
      coherence = as_num(dat$coherence[keep]),
      observation_noise_std = as_num(dat$observation_noise_std[keep]),
      comparison_type = dat$comparison_type[keep],
      parameter_value = dat$parameter_value[keep],
      timestep = step,
      stopping_timestep = stop_time_saved[keep],
      relative_timestep = relative_timestep[keep],
      evidence_sample = evidence_sample[keep],
      cumulative_evidence = cumulative_evidence[keep],
      p_continue = p_continue[keep],
      p_choose_a = p_a[keep],
      p_choose_b = p_b[keep],
      terminal_action = terminal_action[keep],
      action = if (action_col %in% names(dat)) as.integer(round(as_num(dat[[action_col]][keep]))) else NA_integer_,
      choice_direction = choice_direction[keep],
      choice_aligned_cumulative_evidence = choice_direction[keep] * cumulative_evidence[keep],
      p_stop = p_stop[keep],
      p_eventual_choice = p_eventual_choice[keep],
      p_eventual_choice_given_stop = p_eventual_choice_given_stop[keep],
      forced_terminal = forced_terminal[keep],
      explicit_stop = explicit_stop[keep],
      stringsAsFactors = FALSE
    )
  }
  if (length(pieces) == 0L) {
    stop("No valid trajectory timesteps could be reconstructed.")
  }
  out <- do.call(rbind, pieces)
  attr(out, "stopping_discrepancy_count") <- discrepancy_count
  attr(out, "max_policy_norm_error") <- if (length(p_norm_errors) > 0L) max(p_norm_errors, na.rm = TRUE) else NA_real_
  attr(out, "min_p_stop") <- if (length(p_stop_values) > 0L) min(p_stop_values, na.rm = TRUE) else NA_real_
  attr(out, "stabilized_count") <- stabilized_count
  out
}

long_data <- reshape_trajectory_arrays(trial_data, max_steps_before_stop = max_steps_before_stop)
message(sprintf("Number of valid trajectory timesteps before forced-stop filtering: %d", nrow(long_data)))
message(sprintf("Stopping timestep/num_observations discrepancy count: %d", attr(long_data, "stopping_discrepancy_count")))
message(sprintf("Maximum policy-probability normalization error: %.6g", attr(long_data, "max_policy_norm_error")))
message(sprintf("Minimum p_stop: %.6g", attr(long_data, "min_p_stop")))
message(sprintf("Rows using p_stop numerical stabilization (<1e-12): %d", attr(long_data, "stabilized_count")))
if (is.finite(attr(long_data, "max_policy_norm_error")) && attr(long_data, "max_policy_norm_error") > 1e-5) {
  warning(sprintf("Action probabilities deviate from sum 1 by up to %.6g.", attr(long_data, "max_policy_norm_error")))
}
if (is.finite(attr(long_data, "min_p_stop")) && attr(long_data, "min_p_stop") < 1e-6) {
  message(sprintf("Rows with p_stop < 1e-6: %d", sum(long_data$p_stop < 1e-6, na.rm = TRUE)))
}

trial_level <- long_data[!duplicated(long_data$trial_id), , drop = FALSE]
forced_n <- sum(trial_level$forced_terminal, na.rm = TRUE)
message(sprintf(
  "Forced-terminal trials: %d/%d (%.2f%%).",
  forced_n,
  nrow(trial_level),
  100 * forced_n / max(nrow(trial_level), 1)
))
if (!include_forced_stops) {
  long_data <- long_data[!long_data$forced_terminal, , drop = FALSE]
  if (nrow(long_data) == 0L) {
    stop("No valid voluntary stopping trials remain after excluding forced terminal decisions.")
  }
}
message(sprintf("Forced terminal decisions %s.", if (include_forced_stops) "included" else "excluded"))
message(sprintf("Number of valid trajectory timesteps plotted: %d", nrow(long_data)))
message(sprintf("Stopping time range in plotted data: %s to %s",
  num_label(min(long_data$stopping_timestep, na.rm = TRUE)),
  num_label(max(long_data$stopping_timestep, na.rm = TRUE))
))
message(sprintf("Trials retained in beta column: %d", length(unique(long_data$trial_id[long_data$comparison_type == "vary_beta"]))))
message(sprintf("Trials retained in opportunity column: %d", length(unique(long_data$trial_id[long_data$comparison_type == "vary_opportunity"]))))

forced_n_by_condition <- aggregate(
  forced_terminal ~ comparison_type + parameter_value,
  data = trial_level,
  FUN = function(x) sum(x, na.rm = TRUE)
)
total_n_by_condition <- aggregate(
  trial_id ~ comparison_type + parameter_value,
  data = trial_level,
  FUN = length
)
if (nrow(forced_n_by_condition) > 0L) {
  forced_summary <- merge(forced_n_by_condition, total_n_by_condition, by = c("comparison_type", "parameter_value"), all = TRUE)
  names(forced_summary)[names(forced_summary) == "forced_terminal"] <- "forced_n"
  names(forced_summary)[names(forced_summary) == "trial_id"] <- "total_n"
  for (i in seq_len(nrow(forced_summary))) {
    message(sprintf(
      "Forced-terminal %s param=%s: %d/%d (%.2f%%)",
      forced_summary$comparison_type[[i]],
      num_label(forced_summary$parameter_value[[i]]),
      forced_summary$forced_n[[i]],
      forced_summary$total_n[[i]],
      100 * forced_summary$forced_n[[i]] / max(forced_summary$total_n[[i]], 1)
    ))
  }
}

make_metric_long <- function(dat) {
  metrics <- list(
    choice_aligned_cumulative_evidence = "Choice-aligned cumulative evidence",
    p_stop = "Probability of stopping",
    p_eventual_choice_given_stop = "Probability of eventual choice | stop"
  )
  pieces <- vector("list", length(metrics))
  i <- 0L
  for (metric_col in names(metrics)) {
    i <- i + 1L
    pieces[[i]] <- data.frame(
      comparison_type = dat$comparison_type,
      parameter_value = dat$parameter_value,
      run_id = dat$run_id,
      trial_id = dat$trial_id,
      beta = dat$beta,
      opportunity_cost = dat$opportunity_cost,
      observation_noise_std = dat$observation_noise_std,
      relative_timestep = dat$relative_timestep,
      metric = metric_col,
      metric_label = metrics[[metric_col]],
      value = dat[[metric_col]],
      stringsAsFactors = FALSE
    )
  }
  do.call(rbind, pieces)
}

metric_data <- make_metric_long(long_data)
metric_data <- metric_data[is.finite(metric_data$value), , drop = FALSE]

summarize_within_runs <- function(dat) {
  if (requireNamespace("data.table", quietly = TRUE)) {
    dt <- data.table::as.data.table(dat)
    out <- dt[, .(
      run_mean = mean(value, na.rm = TRUE),
      n_trials = data.table::uniqueN(trial_id)
    ), by = .(comparison_type, parameter_value, observation_noise_std, run_id, relative_timestep, metric, metric_label)]
    return(as.data.frame(out))
  }
  aggregate_formula <- value ~ comparison_type + parameter_value + observation_noise_std + run_id + relative_timestep + metric + metric_label
  means <- aggregate(aggregate_formula, data = dat, FUN = function(x) mean(x, na.rm = TRUE))
  names(means)[names(means) == "value"] <- "run_mean"
  counts <- aggregate(
    trial_id ~ comparison_type + parameter_value + observation_noise_std + run_id + relative_timestep + metric + metric_label,
    data = dat,
    FUN = function(x) length(unique(x))
  )
  names(counts)[names(counts) == "trial_id"] <- "n_trials"
  merge(means, counts, by = c("comparison_type", "parameter_value", "observation_noise_std", "run_id", "relative_timestep", "metric", "metric_label"), all.x = TRUE)
}

run_summary <- summarize_within_runs(metric_data)

summarize_across_runs <- function(run_summary) {
  if (requireNamespace("data.table", quietly = TRUE)) {
    dt <- data.table::as.data.table(run_summary)
    out <- dt[, .(
      mean = mean(run_mean, na.rm = TRUE),
      sd_across_runs = if (.N > 1L) stats::sd(run_mean, na.rm = TRUE) else NA_real_,
      n_runs = data.table::uniqueN(run_id),
      n_trials_contributing = sum(n_trials, na.rm = TRUE)
    ), by = .(comparison_type, parameter_value, observation_noise_std, relative_timestep, metric, metric_label)]
    out[, se := ifelse(n_runs > 1L, sd_across_runs / sqrt(n_runs), NA_real_)]
    return(as.data.frame(out))
  }
  group_cols <- c("comparison_type", "parameter_value", "observation_noise_std", "relative_timestep", "metric", "metric_label")
  means <- aggregate(run_mean ~ ., data = run_summary[, c(group_cols, "run_mean"), drop = FALSE], FUN = mean)
  names(means)[names(means) == "run_mean"] <- "mean"
  sds <- aggregate(run_mean ~ ., data = run_summary[, c(group_cols, "run_mean"), drop = FALSE], FUN = function(x) if (length(x) > 1L) stats::sd(x) else NA_real_)
  names(sds)[names(sds) == "run_mean"] <- "sd_across_runs"
  runs <- aggregate(run_id ~ ., data = run_summary[, c(group_cols, "run_id"), drop = FALSE], FUN = function(x) length(unique(x)))
  names(runs)[names(runs) == "run_id"] <- "n_runs"
  trials <- aggregate(n_trials ~ ., data = run_summary[, c(group_cols, "n_trials"), drop = FALSE], FUN = sum)
  names(trials)[names(trials) == "n_trials"] <- "n_trials_contributing"
  out <- Reduce(function(x, y) merge(x, y, by = group_cols, all = TRUE), list(means, sds, runs, trials))
  out$se <- ifelse(out$n_runs > 1L, out$sd_across_runs / sqrt(out$n_runs), NA_real_)
  out
}

plot_summary <- summarize_across_runs(run_summary)
plot_summary <- plot_summary[plot_summary$n_trials_contributing >= minimum_samples, , drop = FALSE]
if (nrow(plot_summary) == 0L) {
  stop("No summarized points remain after applying --min-samples.")
}
plot_summary$fixed_beta <- fixed_beta
plot_summary$fixed_opp <- fixed_opp
plot_summary$fixed_coherence <- fixed_coherence
plot_summary$fixed_observation_noise_std <- plot_summary$observation_noise_std
plot_summary$pay_kl_on_stop <- pay_kl_on_stop_mode
plot_summary$include_forced_stops <- include_forced_stops
plot_summary$parameter_label <- ifelse(
  plot_summary$comparison_type == "vary_beta",
  paste0("beta ", vapply(plot_summary$parameter_value, num_label, character(1))),
  paste0("opp ", vapply(plot_summary$parameter_value, num_label, character(1)))
)

for (metric_name in unique(plot_summary$metric)) {
  for (comparison in unique(plot_summary$comparison_type)) {
    params <- sort(unique(plot_summary$parameter_value[plot_summary$metric == metric_name & plot_summary$comparison_type == comparison]))
    for (obsstd in selected_obsstd_values) {
      for (param in params) {
        rows <- plot_summary$metric == metric_name &
          plot_summary$comparison_type == comparison &
          parameter_equal(plot_summary$observation_noise_std, obsstd) &
          parameter_equal(plot_summary$parameter_value, param)
        n_trials <- plot_summary$n_trials_contributing[rows]
        if (length(n_trials) == 0L) next
        message(sprintf(
          "Contributing trials for %s std=%s %s param=%s across relative time: min=%d max=%d",
          metric_name,
          num_label(obsstd),
          comparison,
          num_label(param),
          min(n_trials, na.rm = TRUE),
          max(n_trials, na.rm = TRUE)
        ))
      }
    }
  }
}

plot_font_size_pt <- 7
line_width <- 1.3
ribbon_alpha <- 0.18
target_panel_side_mm <- 33
target_panel_side_in <- target_panel_side_mm / 25.4
panel_margin_in <- c(bottom = 0.66, left = 0.78, top = 0.08, right = 0.12)
label_margin_in <- c(bottom = 0.02, left = 0.02, top = 0.02, right = 0.02)

family_color_values <- function(family, params) {
  params <- sort(unique(as_num(params)))
  params <- params[is.finite(params)]
  if (length(params) == 0L) return(character())
  palette <- if (identical(family, "vary_beta")) {
    grDevices::colorRampPalette(c("#00441b", "#238b45", "#74c476"))
  } else {
    grDevices::colorRampPalette(c("#6baed6", "#2171b5", "#08306b"))
  }
  cols <- palette(max(length(params), 2L))[seq_along(params)]
  names(cols) <- as.character(params)
  cols
}

series_pch <- function(family) if (identical(family, "vary_beta")) 16 else 17

series_color <- function(family, parameter_value) {
  vals <- if (identical(family, "vary_beta")) beta_values else opportunity_values
  colors <- family_color_values(family, vals)
  colors[[as.character(as_num(parameter_value))]]
}

safe_ylim <- function(values, se = NULL, pad_fraction = 0.08, fallback = c(0, 1)) {
  candidates <- c(values)
  if (!is.null(se)) {
    candidates <- c(candidates, values - se, values + se)
  }
  candidates <- as_num(candidates)
  candidates <- candidates[is.finite(candidates)]
  if (length(candidates) == 0L) return(fallback)
  lim <- range(candidates)
  if (abs(diff(lim)) < 1e-12) lim <- lim + c(-0.5, 0.5)
  pad <- diff(lim) * pad_fraction
  lim + c(-pad, pad)
}

metric_ylim <- function(metric_name) {
  metric_rows <- plot_summary$metric == metric_name
  values <- plot_summary$mean[metric_rows]
  se <- plot_summary$se[metric_rows]
  if (identical(metric_name, "p_stop")) {
    return(c(0, 1))
  }
  if (identical(metric_name, "p_eventual_choice_given_stop")) {
    valid <- as_num(values)
    valid <- valid[is.finite(valid)]
    if (length(valid) > 0L && min(valid) >= 0.5) {
      return(c(0.5, 1))
    }
    return(c(0, 1))
  }
  lim <- safe_ylim(values, se)
  if (lim[1L] > 0) lim[1L] <- min(0, lim[1L])
  lim
}

metric_specs <- list(
  choice_aligned_cumulative_evidence = list(
    slug = "choice_aligned_cumulative_evidence",
    xlab = "Steps relative\nto stopping",
    ylab = "Choice-aligned\ncumulative evidence",
    hline = 0
  ),
  p_stop = list(
    slug = "stopping_probability",
    xlab = "Steps relative\nto stopping",
    ylab = "Probability\nof stopping",
    hline = 0.5
  ),
  p_eventual_choice_given_stop = list(
    slug = "eventual_choice_given_stop",
    xlab = "Steps relative\nto stopping",
    ylab = "P(eventual choice)\nconditional on stop",
    hline = 0.5
  )
)

draw_response_panel <- function(panel_data, family, metric_name, spec) {
  old_xpd <- par("xpd")
  on.exit(par(xpd = old_xpd), add = TRUE)
  par(xpd = FALSE)
  y_lim <- metric_ylim(metric_name)
  x_lim <- c(-max_steps_before_stop, 0)
  plot(NA,
    xlim = x_lim,
    ylim = y_lim,
    xlab = spec$xlab,
    ylab = spec$ylab,
    xaxt = "n",
    cex.lab = 1,
    cex.axis = 1
  )
  axis(1, at = seq.int(-max_steps_before_stop, 0L, by = 1L))
  grid(col = "grey90")
  abline(v = 0, col = "grey55", lty = 2, lwd = 0.8)
  if (!is.null(spec$hline)) {
    abline(h = spec$hline, col = "grey55", lty = 2, lwd = 0.8)
  }
  params <- sort(unique(as_num(panel_data$parameter_value)))
  for (param in params) {
    line_data <- panel_data[parameter_equal(panel_data$parameter_value, param), , drop = FALSE]
    line_data <- line_data[order(line_data$relative_timestep), , drop = FALSE]
    if (nrow(line_data) == 0L) next
    col <- series_color(family, param)
    x <- as_num(line_data$relative_timestep)
    y <- as_num(line_data$mean)
    se <- as_num(line_data$se)
    ribbon_keep <- is.finite(x) & is.finite(y) & is.finite(se) & se > 0
    if (sum(ribbon_keep) >= 2L) {
      xr <- x[ribbon_keep]
      yr <- y[ribbon_keep]
      ser <- se[ribbon_keep]
      polygon(
        c(xr, rev(xr)),
        c(yr - ser, rev(yr + ser)),
        col = grDevices::adjustcolor(col, alpha.f = ribbon_alpha),
        border = NA
      )
    }
    lines(x, y, col = col, lwd = line_width)
    points(x, y, col = col, pch = series_pch(family), cex = 0.65)
  }
  box()
}

draw_family_header <- function(family, params, colors, header, legend_title) {
  plot.new()
  text(0.5, 0.82, header, cex = 1)
  params <- sort(unique(as_num(params)))
  params <- params[is.finite(params)]
  if (length(params) == 0L) return(invisible(NULL))
  labels <- if (identical(family, "vary_beta")) {
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
    lwd = line_width,
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

beta_colors <- family_color_values("vary_beta", beta_values)
opp_colors <- family_color_values("vary_opportunity", opportunity_values)
obsstd_levels <- sort(unique(as_num(selected_obsstd_values)))
obsstd_levels <- obsstd_levels[is.finite(obsstd_levels)]

output_file_stem <- if (!is.null(output_file_option$value) && nzchar(trim_string(output_file_option$value))) {
  sub("\\.png$", "", output_file_option$value, ignore.case = TRUE)
} else {
  out_dir <- file.path(output_root, "evidence_accumulation_compare")
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  file.path(
    out_dir,
    sprintf(
      "evidence_accumulation_response_locked_beta_opp_obsstd_%s%s",
      values_token(obsstd_levels),
      if (pay_kl_on_stop_mode) "_stop_paid" else ""
    )
  )
}
output_dir <- dirname(output_file_stem)
if (!nzchar(output_dir) || identical(output_dir, ".")) {
  output_dir <- "."
}
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
summary_file <- sprintf("%s_summary.csv", output_file_stem)

plot_file_for_metric <- function(spec) {
  sprintf("%s_%s.png", output_file_stem, spec$slug)
}

save_metric_plot <- function(metric_name) {
  spec <- metric_specs[[metric_name]]
  n_plot_rows <- length(obsstd_levels)
  next_layout_id <- 1L
  legend_beta_id <- next_layout_id
  next_layout_id <- next_layout_id + 1L
  legend_opp_id <- next_layout_id
  next_layout_id <- next_layout_id + 1L
  layout_matrix <- matrix(0L, nrow = n_plot_rows + 1L, ncol = 3L)
  layout_matrix[1L, ] <- c(0L, legend_beta_id, legend_opp_id)
  for (row_i in seq_len(n_plot_rows)) {
    layout_matrix[row_i + 1L, ] <- c(next_layout_id, next_layout_id + 1L, next_layout_id + 2L)
    next_layout_id <- next_layout_id + 3L
  }
  label_col_width_in <- 0.42
  header_row_height_in <- 0.9
  panel_cell_width_in <- target_panel_side_in + panel_margin_in[["left"]] + panel_margin_in[["right"]]
  panel_cell_height_in <- target_panel_side_in + panel_margin_in[["bottom"]] + panel_margin_in[["top"]]
  device_width_in <- label_col_width_in + 2 * panel_cell_width_in
  device_height_in <- header_row_height_in + n_plot_rows * panel_cell_height_in
  output_file <- plot_file_for_metric(spec)
  grDevices::png(
    output_file,
    width = device_width_in,
    height = device_height_in,
    units = "in",
    res = 300,
    pointsize = plot_font_size_pt,
    bg = "white"
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
    "vary_beta",
    beta_values,
    beta_colors,
    sprintf("Varying beta\nopp = %s", num_label(fixed_opp)),
    expression(beta)
  )
  par(mai = c(0.02, 0.02, 0.02, 0.08))
  draw_family_header(
    "vary_opportunity",
    opportunity_values,
    opp_colors,
    sprintf("Varying opportunity cost\nbeta = %s", num_label(fixed_beta)),
    "Opportunity cost"
  )
  for (row_i in seq_along(obsstd_levels)) {
    obsstd <- obsstd_levels[[row_i]]
    metric_rows <- plot_summary$metric == metric_name & parameter_equal(plot_summary$observation_noise_std, obsstd)
    metric_data_for_std <- plot_summary[metric_rows, , drop = FALSE]
    par(mai = label_margin_in)
    draw_obsstd_label(obsstd)
    par(mai = panel_margin_in)
    draw_response_panel(
      metric_data_for_std[metric_data_for_std$comparison_type == "vary_beta", , drop = FALSE],
      "vary_beta",
      metric_name,
      spec
    )
    par(mai = panel_margin_in)
    draw_response_panel(
      metric_data_for_std[metric_data_for_std$comparison_type == "vary_opportunity", , drop = FALSE],
      "vary_opportunity",
      metric_name,
      spec
    )
  }
  invisible(try(par(old_par), silent = TRUE))
  grDevices::dev.off()
  message(sprintf("Saved response-locked PNG to: %s", output_file))
  invisible(output_file)
}

metric_order <- c("choice_aligned_cumulative_evidence", "p_stop", "p_eventual_choice_given_stop")
invisible(lapply(metric_order, save_metric_plot))

summary_out <- plot_summary[, c(
  "comparison_type", "parameter_value", "parameter_label", "fixed_beta", "fixed_opp",
  "fixed_coherence", "observation_noise_std", "fixed_observation_noise_std", "relative_timestep", "metric",
  "mean", "se", "n_runs", "n_trials_contributing", "pay_kl_on_stop", "include_forced_stops"
), drop = FALSE]
utils::write.csv(summary_out, summary_file, row.names = FALSE)

message(sprintf("Saved response-locked summary CSV to: %s", summary_file))
message("Stopping alignment: relative_timestep = timestep - num_observations, so tau=0 is the terminal policy decision after the final observed evidence sample.")
message("Choice-aligned evidence: cumulative_evidence_t multiplied by +1 for CHOOSE_B and -1 for CHOOSE_A.")
message("Stopping probability: policy_choose_a_t + policy_choose_b_t.")
message("Conditional choice certainty: probability assigned to the eventual terminal action divided by stopping probability.")
message("Aggregation: trial means are first averaged within independent run/file, then summarized across runs with SE ribbons.")
