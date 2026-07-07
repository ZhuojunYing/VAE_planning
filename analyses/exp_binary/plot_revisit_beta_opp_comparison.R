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
    "  Rscript analyses/exp_binary/plot_revisit_beta_opp_comparison.R <tree> [options]\n\n",
    "Preset trees are read from analyses/exp_binary/revisit_plot_presets.csv.\n",
    "The script loads the <tree>,beta row with opportunity=0 and the <tree>,opportunity row with beta=1000,\n",
    "then plots the requested revisit behavior diagnostics in one comparison folder.\n\n",
    "Options:\n",
    "  --preset-file PATH          Preset CSV path.\n",
    "  --vary-beta-values LIST     Beta values for the beta-vary family; all other params come from the preset.\n",
    "                              Alias: --beta-values, --betas, --vary-betas.\n",
    "  --vary-opportunity-values LIST\n",
    "                              Opportunity costs for the opportunity-vary family; all other params come from the preset.\n",
    "                              Alias: --opportunity-values, --opportunities, --opportunity-costs, --vary-opps.\n",
    "  --sigmas LIST               Override sigma values.\n",
    "  --seeds LIST                Override seed values.\n",
    "  --output-root DIR           Override output root. Default is preset results_dir.\n",
    "  --input-dir DIR             Override simulation input dir.\n",
    "  --sampled-lambda-critic q|value\n",
    "                              Simulation filename mode. q is default and matches legacy files without _vcritic.\n",
    "                              value/v matches files with the _vcritic suffix. Aliases: --critic, --critic-type.\n",
    "  --min-samples N             Drop dots with fewer than N trial/event samples. Default 10.\n",
    "  --help                      Show this message.\n\n",
    "Example:\n",
    "  Rscript analyses/exp_binary/plot_revisit_beta_opp_comparison.R default \\\n",
    "    --vary-beta-values \"2,4,6,8,10\" \\\n",
    "    --vary-opportunity-values \"0.02,0.04,0.08,0.1\" \\\n",
    "    --min-samples 25\n",
    sep = ""
  )
}

if (length(args) == 0L || any(args %in% c("--help", "-h"))) {
  usage()
  quit(save = "no", status = if (length(args) == 0L) 1L else 0L)
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

min_samples_option <- extract_named_option(
  args,
  c("--min-samples", "--min-sampes", "--min-n"),
  default = "10"
)
args <- min_samples_option$args
minimum_samples <- suppressWarnings(as.integer(round(as.numeric(min_samples_option$value))))
if (is.na(minimum_samples) || minimum_samples < 0L) {
  stop("--min-samples must be a nonnegative integer.")
}

preset_file_option <- extract_named_option(
  args,
  c("--preset-file"),
  default = file.path(script_dir, "revisit_plot_presets.csv")
)
args <- preset_file_option$args

beta_values_option <- extract_named_option(
  args,
  c("--vary-beta-values", "--vary-betas", "--beta-values", "--betas"),
  default = NULL
)
args <- beta_values_option$args
opportunity_values_option <- extract_named_option(
  args,
  c(
    "--vary-opportunity-values",
    "--vary-opportunities",
    "--vary-opps",
    "--opportunity-values",
    "--opportunities",
    "--opportunity-costs"
  ),
  default = NULL
)
args <- opportunity_values_option$args
sigma_values_option <- extract_named_option(args, c("--sigmas", "--sigma-list"), default = NULL)
args <- sigma_values_option$args
seed_values_option <- extract_named_option(args, c("--seeds"), default = NULL)
args <- seed_values_option$args
output_root_option <- extract_named_option(args, c("--output-root", "--results-dir"), default = NULL)
args <- output_root_option$args
input_dir_option <- extract_named_option(args, c("--input-dir"), default = NULL)
args <- input_dir_option$args
critic_option <- extract_named_option(
  args,
  c("--sampled-lambda-critic", "--critic", "--critic-type", "--critic-mode"),
  default = "q"
)
args <- critic_option$args

if (length(args) != 1L) {
  usage()
  stop("Expected exactly one positional argument: <tree>.")
}

tree_arg <- trim_string(args[[1]])

parse_csv_values <- function(value) {
  if (is.null(value) || !nzchar(trim_string(value))) {
    return(character())
  }
  out <- unlist(strsplit(as.character(value), ",", fixed = TRUE), use.names = FALSE)
  out <- trimws(out)
  out[nzchar(out)]
}

num_tokens <- function(value) {
  raw <- trim_string(value)
  numeric_value <- suppressWarnings(as.numeric(raw))
  tokens <- raw
  if (is.finite(numeric_value)) {
    tokens <- c(
      tokens,
      format(numeric_value, scientific = FALSE, trim = TRUE),
      sprintf("%.1f", numeric_value),
      sub("\\.0$", "", sprintf("%.1f", numeric_value))
    )
  }
  unique(tokens[nzchar(tokens)])
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
    stop(sprintf("--sampled-lambda-critic must be q or value/v. Got %s.", value))
  }
  unname(aliases[[key]])
}

sampled_lambda_critic <- normalize_sampled_lambda_critic(critic_option$value)

sampled_lambda_critic_file_suffixes <- function() {
  if (identical(sampled_lambda_critic, "value")) "_vcritic" else c("", "_qcritic")
}

visited_lstm_suffix_variants <- function(suffixes) {
  unique(c(suffixes, paste0(suffixes, "_visitedidx")))
}

parsed_revisit_file_index <- NULL

parse_revisit_filename_index <- function() {
  files <- list.files(input_dir, pattern = "_revisit_maxobs_[0-9]+.*_uniform\\.csv$", full.names = TRUE)
  if (length(files) == 0L) {
    return(data.frame())
  }
  basenames <- basename(files)
  pattern <- paste0(
    "^lambda_([^_]+)_alpha_([^_]+)_beta_([^_]+)_opportunity_([^_]+)",
    "_expansion_([^_]+)_variant_([^_]+)_seed_([0-9]+)_(.+)_rnn_([^_]+)_latent_([^_]+)",
    "_revisit_maxobs_([0-9]+)(?:_obs_sigma_([^_]+))?",
    "(?:_klstart_[^_]+_klanneal_[^_]+)?(_(?:q|v)critic)?(?:_visitedidx)?_uniform\\.csv$"
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
  sigma_token <- part_at(13L)
  sigma_token[!nzchar(sigma_token)] <- "0"
  critic_token <- part_at(14L)
  out <- data.frame(
    file = files,
    lambda = suppressWarnings(as.numeric(part_at(2L))),
    alpha = suppressWarnings(as.numeric(part_at(3L))),
    beta = suppressWarnings(as.numeric(part_at(4L))),
    opportunity = suppressWarnings(as.numeric(part_at(5L))),
    expansion = part_at(6L),
    variant = part_at(7L),
    seed = suppressWarnings(as.numeric(part_at(8L))),
    tree_label = part_at(9L),
    rnn_units = part_at(10L),
    latent_dim = part_at(11L),
    max_observations = part_at(12L),
    sigma = suppressWarnings(as.numeric(sigma_token)),
    critic = ifelse(grepl("vcritic", critic_token, fixed = TRUE), "value", "q"),
    stringsAsFactors = FALSE
  )
  out
}

get_revisit_file_index <- function() {
  if (is.null(parsed_revisit_file_index)) {
    parsed_revisit_file_index <<- parse_revisit_filename_index()
  }
  parsed_revisit_file_index
}

num_label <- function(value) {
  value <- suppressWarnings(as.numeric(value))
  if (!is.finite(value)) {
    return(as.character(value))
  }
  label <- format(value, scientific = FALSE, trim = TRUE, digits = 8)
  if (grepl("\\.", label)) {
    label <- sub("0+$", "", label)
    label <- sub("\\.$", "", label)
  }
  if (!nzchar(label)) "0" else label
}

as_num <- function(value) suppressWarnings(as.numeric(as.character(value)))

parameter_equal <- function(x, y, tol = 1e-8) {
  x <- suppressWarnings(as.numeric(x))
  y <- suppressWarnings(as.numeric(y))
  is.finite(x) & is.finite(y) & abs(x - y) <= tol
}

axis_filename_label <- function(value) {
  value <- gsub("_+", "_", gsub("[^A-Za-z0-9]+", "_", tolower(as.character(value))))
  value <- gsub("^_|_$", "", value)
  if (!nzchar(value)) "plot" else value
}

read_csv_fast <- function(path, select = NULL) {
  if (requireNamespace("data.table", quietly = TRUE)) {
    return(as.data.frame(data.table::fread(path, select = select, showProgress = FALSE)))
  }
  dat <- utils::read.csv(path, check.names = FALSE)
  if (!is.null(select)) {
    dat <- dat[, intersect(select, names(dat)), drop = FALSE]
  }
  dat
}

header_cols <- function(path) {
  if (requireNamespace("data.table", quietly = TRUE)) {
    return(names(data.table::fread(path, nrows = 0L, showProgress = FALSE)))
  }
  names(utils::read.csv(path, nrows = 1L, check.names = FALSE))
}

normalize_tree_name <- function(value) {
  value <- trim_string(value)
  value <- sub("^2n$", "default", value)
  value <- sub("^3n_bandit3$", "bandit3", value)
  value <- sub("^4n_disjoint2x2$", "disjoint2x2", value)
  value <- sub("^6n_disjoint3x2$", "disjoint3x2", value)
  value
}

preset_file <- preset_file_option$value
if (!file.exists(preset_file)) {
  stop(sprintf("Preset file not found: %s", preset_file))
}
preset_data <- utils::read.csv(preset_file, stringsAsFactors = FALSE, check.names = FALSE)
preset_data$tree <- vapply(preset_data$tree, normalize_tree_name, character(1))
tree_name <- normalize_tree_name(tree_arg)
beta_row <- preset_data[preset_data$tree == tree_name & preset_data$vary == "beta", , drop = FALSE]
opp_row <- preset_data[preset_data$tree == tree_name & preset_data$vary == "opportunity", , drop = FALSE]
if (nrow(beta_row) == 0L || nrow(opp_row) == 0L) {
  stop(sprintf("Need both beta and opportunity rows for tree=%s in %s.", tree_name, preset_file))
}
beta_row <- beta_row[1L, , drop = FALSE]
opp_row <- opp_row[1L, , drop = FALSE]

shared <- beta_row
tree_size <- as.integer(shared$tree_size[[1]])
tree_config <- trim_string(shared$tree_config[[1]])
if (!nzchar(tree_config)) {
  tree_config <- "default"
}
input_type <- trim_string(shared$input_type[[1]])
expansion_decision_version <- trim_string(shared$expansion_decision_version[[1]])
model_variant <- trim_string(shared$model_variant[[1]])
rnn_units <- trim_string(shared$rnn_units_arg[[1]])
latent_dim <- trim_string(shared$latent_dim_arg[[1]])
lambda_arg <- trim_string(shared$lambda_arg[[1]])
alpha_arg <- trim_string(shared$alpha_arg[[1]])
source_arg <- trim_string(shared$simulation_source_arg[[1]])
max_observations <- trim_string(shared$max_observations_arg[[1]])
seed_values <- if (!is.null(seed_values_option$value)) parse_csv_values(seed_values_option$value) else parse_csv_values(shared$seed_arg[[1]])
sigma_values <- if (!is.null(sigma_values_option$value)) parse_csv_values(sigma_values_option$value) else parse_csv_values(shared$sigma_arg[[1]])
beta_values <- if (!is.null(beta_values_option$value)) parse_csv_values(beta_values_option$value) else parse_csv_values(beta_row$beta_arg[[1]])
opportunity_values <- if (!is.null(opportunity_values_option$value)) parse_csv_values(opportunity_values_option$value) else parse_csv_values(opp_row$opportunity_arg[[1]])
beta_family_opportunity <- trim_string(beta_row$opportunity_arg[[1]])
opportunity_family_beta <- trim_string(opp_row$beta_arg[[1]])
base_input_dir <- if (!is.null(input_dir_option$value)) input_dir_option$value else shared$input_dir[[1]]
input_dir <- if (tolower(source_arg) == "jax" && basename(base_input_dir) == "simulations") {
  file.path(dirname(base_input_dir), "jax_simulations")
} else {
  base_input_dir
}
output_root <- if (!is.null(output_root_option$value)) output_root_option$value else shared$results_dir[[1]]

if (!dir.exists(input_dir)) {
  stop(sprintf("Simulation input directory not found: %s", input_dir))
}
message(sprintf(
  "Sampled-lambda critic file mode: %s%s",
  sampled_lambda_critic,
  if (identical(sampled_lambda_critic, "q")) " (legacy/no _vcritic suffix)" else " (_vcritic suffix)"
))

tree_label <- if (tree_config %in% c("", "default")) {
  sprintf("%dn", tree_size)
} else {
  sprintf("%dn_%s", tree_size, tree_config)
}

reward_values <- if (identical(input_type, "binary")) c(0, 1) else c(-4, -3, -2, -1, 1, 2, 3, 4)
is_disjoint2x2 <- identical(tree_config, "disjoint2x2")
is_disjoint3x2 <- identical(tree_config, "disjoint3x2")
task_nodes_per_path <- if (is_disjoint2x2 || is_disjoint3x2) 2L else 1L
task_path_count <- if (is_disjoint3x2) {
  3L
} else if (is_disjoint2x2) {
  2L
} else {
  tree_size
}
task_node_count <- task_path_count * task_nodes_per_path

task_path_nodes <- function() {
  split(seq_len(task_node_count), rep(seq_len(task_path_count), each = task_nodes_per_path))
}

path_id_for_node <- function(node) {
  node_num <- suppressWarnings(as.integer(node))
  out <- rep(NA_integer_, length(node_num))
  valid <- !is.na(node_num) & node_num >= 1L
  if (is_disjoint2x2 || is_disjoint3x2) {
    out[valid] <- floor((node_num[valid] - 1L) / 2L) + 1L
  } else {
    out[valid] <- node_num[valid]
  }
  out
}

expected_best_path_reward <- function() {
  grids <- expand.grid(rep(list(reward_values), task_node_count))
  path_nodes <- task_path_nodes()
  path_rewards <- vapply(path_nodes, function(nodes) {
    rowSums(grids[, nodes, drop = FALSE])
  }, numeric(nrow(grids)))
  mean(apply(path_rewards, 1L, max))
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

column_timesteps <- function(names_vec, prefix) {
  cols <- grep(paste0("^", prefix, "[0-9]+$"), names_vec, value = TRUE)
  timesteps <- suppressWarnings(as.integer(sub(prefix, "", cols)))
  timesteps <- sort(timesteps[is.finite(timesteps)])
  timesteps
}

numeric_matrix_from_cols <- function(dat, cols) {
  if (length(cols) == 0L) {
    return(matrix(numeric(), nrow = nrow(dat), ncol = 0L))
  }
  out <- as.matrix(data.frame(lapply(dat[, cols, drop = FALSE], function(x) suppressWarnings(as.numeric(x))), check.names = FALSE))
  storage.mode(out) <- "numeric"
  out
}

logical_matrix_from_cols <- function(dat, cols) {
  if (length(cols) == 0L) {
    return(matrix(logical(), nrow = nrow(dat), ncol = 0L))
  }
  as.matrix(data.frame(lapply(dat[, cols, drop = FALSE], as_logical_col), check.names = FALSE))
}

path_reward_matrix_from_node_rewards <- function(actual_reward_mat) {
  if (nrow(actual_reward_mat) == 0L) {
    return(matrix(numeric(), nrow = 0L, ncol = task_path_count))
  }
  path_nodes <- task_path_nodes()
  out <- vapply(path_nodes, function(nodes) {
    rowSums(actual_reward_mat[, nodes, drop = FALSE])
  }, numeric(nrow(actual_reward_mat)))
  if (is.null(dim(out))) {
    out <- matrix(out, nrow = nrow(actual_reward_mat), ncol = task_path_count)
  }
  out
}

realized_reward_denominator <- function(path_reward_mat) {
  if (nrow(path_reward_mat) == 0L || ncol(path_reward_mat) == 0L) {
    return(task_reward_norm)
  }
  finite_rows <- rowSums(is.finite(path_reward_mat)) == ncol(path_reward_mat)
  if (!any(finite_rows)) {
    return(task_reward_norm)
  }
  denom <- mean(apply(path_reward_mat[finite_rows, , drop = FALSE], 1L, max), na.rm = TRUE)
  if (is.finite(denom) && abs(denom) > 1e-12) {
    denom
  } else {
    task_reward_norm
  }
}

actual_path_reward_tie_flags <- function(path_reward_mat, tol = 1e-8) {
  out <- rep(FALSE, nrow(path_reward_mat))
  finite_rows <- rowSums(is.finite(path_reward_mat)) == ncol(path_reward_mat)
  if (!any(finite_rows) || ncol(path_reward_mat) < 2L) {
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

first_finite_col <- function(mat) {
  if (ncol(mat) == 0L) {
    return(rep(NA_integer_, nrow(mat)))
  }
  finite_mat <- is.finite(mat)
  idx <- max.col(ifelse(finite_mat, ncol(mat) - col(mat) + 1L, 0), ties.method = "first")
  idx[rowSums(finite_mat) == 0L] <- NA_integer_
  idx
}

node_actual_reward_matrix <- function(dat, trial_data) {
  out <- matrix(NA_real_, nrow = nrow(trial_data), ncol = task_node_count)
  if (!all(c("node", "actual_reward") %in% names(dat))) {
    return(out)
  }
  if ("graph" %in% names(dat) && "graph" %in% names(trial_data)) {
    graph_keys <- as.character(trial_data$graph)
    for (node_id in seq_len(task_node_count)) {
      node_rows <- dat[suppressWarnings(as.integer(dat$node)) == node_id, c("graph", "actual_reward"), drop = FALSE]
      if (nrow(node_rows) == 0L) {
        next
      }
      node_rows <- node_rows[!duplicated(node_rows$graph), , drop = FALSE]
      out[, node_id] <- suppressWarnings(as.numeric(node_rows$actual_reward[match(graph_keys, as.character(node_rows$graph))]))
    }
  }
  out
}

actual_reward_for_observed_node <- function(node_values, actual_reward_mat, fallback_values = NULL) {
  node_values <- suppressWarnings(as.integer(node_values))
  out <- rep(NA_real_, length(node_values))
  valid <- !is.na(node_values) & node_values >= 1L & node_values <= ncol(actual_reward_mat)
  rows <- which(valid)
  if (length(rows) > 0L) {
    out[rows] <- actual_reward_mat[cbind(rows, node_values[rows])]
  }
  if (!is.null(fallback_values)) {
    fallback_values <- suppressWarnings(as.numeric(fallback_values))
    out <- ifelse(is.finite(out), out, fallback_values)
  }
  out
}

terminal_prob_cols_for_timestep <- function(trial_data, timestep) {
  explicit_pattern <- paste0("^terminal_choice_prob_path[0-9]+_t", timestep, "$")
  fallback_pattern <- paste0("^action_output_path[0-9]+_t", timestep, "$")
  cols <- grep(explicit_pattern, names(trial_data), value = TRUE)
  if (length(cols) == 0L) {
    cols <- grep(fallback_pattern, names(trial_data), value = TRUE)
  }
  if (length(cols) == 0L) {
    return(character())
  }
  path_idx <- suppressWarnings(as.integer(sub("^.*_path([0-9]+)_t[0-9]+$", "\\1", cols)))
  cols[order(path_idx)]
}

terminal_binary_choice_entropy_for_timestep <- function(trial_data, timestep, tied_rows) {
  prob_cols <- terminal_prob_cols_for_timestep(trial_data, timestep)
  if (length(prob_cols) < 2L) {
    return(rep(NA_real_, nrow(trial_data)))
  }
  prob_cols <- prob_cols[seq_len(2L)]
  prob_mat <- numeric_matrix_from_cols(trial_data, prob_cols)
  prob_mat[!is.finite(prob_mat) | prob_mat <= 0] <- 0
  prob_sums <- rowSums(prob_mat)
  entropy <- rep(NA_real_, nrow(prob_mat))
  valid <- is.finite(prob_sums) & prob_sums > 0
  if (any(valid)) {
    normalized <- prob_mat[valid, , drop = FALSE] / prob_sums[valid]
    entropy[valid] <- -rowSums(ifelse(normalized > 0, normalized * log(normalized), 0))
  }
  entropy[tied_rows] <- NA_real_
  entropy
}

find_sim_file <- function(lambda_value, alpha_value, beta_value, opportunity_value, seed_value, sigma_value) {
  lambda_tokens <- num_tokens(lambda_value)
  alpha_tokens <- num_tokens(alpha_value)
  beta_tokens <- num_tokens(beta_value)
  opportunity_tokens <- num_tokens(opportunity_value)
  seed_tokens <- num_tokens(seed_value)
  sigma_tokens <- num_tokens(sigma_value)
  candidates <- character()
  for (lambda_token in lambda_tokens) {
    for (alpha_token in alpha_tokens) {
      for (beta_token in beta_tokens) {
        for (opportunity_token in opportunity_tokens) {
          for (seed_token in seed_tokens) {
            base <- sprintf(
              "lambda_%s_alpha_%s_beta_%s_opportunity_%s_expansion_%s_variant_%s_seed_%s_%s_rnn_%s_latent_%s_revisit_maxobs_%s",
              lambda_token,
              alpha_token,
              beta_token,
              opportunity_token,
              expansion_decision_version,
              model_variant,
              seed_token,
              tree_label,
              rnn_units,
              latent_dim,
              max_observations
            )
            sigma_num <- suppressWarnings(as.numeric(sigma_value))
            for (critic_suffix in sampled_lambda_critic_file_suffixes()) {
              if (is.finite(sigma_num) && abs(sigma_num) < 1e-12) {
                suffixes <- visited_lstm_suffix_variants(c(
                  critic_suffix,
                  paste0("_obs_sigma_0", critic_suffix),
                  paste0("_obs_sigma_0.0", critic_suffix)
                ))
                candidates <- c(candidates, file.path(input_dir, paste0(base, suffixes, "_", input_type, ".csv")))
              } else {
                for (sigma_token in sigma_tokens) {
                  suffixes <- visited_lstm_suffix_variants(paste0("_obs_sigma_", sigma_token, critic_suffix))
                  candidates <- c(candidates, file.path(input_dir, paste0(base, suffixes, "_", input_type, ".csv")))
                }
              }
            }
          }
        }
      }
    }
  }
  candidates <- unique(candidates)
  found <- candidates[file.exists(candidates)]
  if (length(found) > 0L) {
    return(found[[1L]])
  }
  index <- get_revisit_file_index()
  if (nrow(index) > 0L) {
    lambda_num <- as_num(lambda_value)
    alpha_num <- as_num(alpha_value)
    beta_num <- as_num(beta_value)
    opportunity_num <- as_num(opportunity_value)
    seed_num <- as_num(seed_value)
    sigma_num <- as_num(sigma_value)
    tol <- 1e-8
    matches <- is.finite(index$lambda) & abs(index$lambda - lambda_num) <= tol &
      is.finite(index$alpha) & abs(index$alpha - alpha_num) <= tol &
      is.finite(index$beta) & abs(index$beta - beta_num) <= tol &
      is.finite(index$opportunity) & abs(index$opportunity - opportunity_num) <= tol &
      is.finite(index$seed) & abs(index$seed - seed_num) <= tol &
      is.finite(index$sigma) & abs(index$sigma - sigma_num) <= tol &
      index$expansion == expansion_decision_version &
      index$variant == model_variant &
      index$tree_label == tree_label &
      index$rnn_units == rnn_units &
      index$latent_dim == latent_dim &
      index$max_observations == max_observations &
      index$critic == sampled_lambda_critic
    if (any(matches)) {
      return(index$file[which(matches)[[1L]]])
    }
  }
  NA_character_
}

build_file_manifest <- function() {
  rows <- list()
  missing <- 0L
  add_family <- function(family, parameter_values, beta_value_fun, opportunity_value_fun) {
    local_rows <- list()
    local_missing <- 0L
    for (parameter_value in parameter_values) {
      beta_value <- beta_value_fun(parameter_value)
      opportunity_value <- opportunity_value_fun(parameter_value)
      for (seed_value in seed_values) {
        for (sigma_value in sigma_values) {
          path <- find_sim_file(lambda_arg, alpha_arg, beta_value, opportunity_value, seed_value, sigma_value)
          if (is.na(path)) {
            local_missing <- local_missing + 1L
            next
          }
          local_rows[[length(local_rows) + 1L]] <- data.frame(
            family = family,
            parameter_value = as_num(parameter_value),
            parameter_label = if (family == "beta") paste0("beta=", num_label(parameter_value)) else paste0("opp=", num_label(parameter_value)),
            beta = as_num(beta_value),
            opportunity = as_num(opportunity_value),
            seed = as_num(seed_value),
            sigma = as_num(sigma_value),
            file = path,
            stringsAsFactors = FALSE
          )
        }
      }
    }
    list(rows = local_rows, missing = local_missing)
  }
  beta_manifest <- add_family(
    "beta",
    beta_values,
    beta_value_fun = function(parameter_value) parameter_value,
    opportunity_value_fun = function(parameter_value) beta_family_opportunity
  )
  opp_manifest <- add_family(
    "opportunity",
    opportunity_values,
    beta_value_fun = function(parameter_value) opportunity_family_beta,
    opportunity_value_fun = function(parameter_value) parameter_value
  )
  rows <- c(beta_manifest$rows, opp_manifest$rows)
  missing <- beta_manifest$missing + opp_manifest$missing
  if (length(rows) == 0L) {
    stop("No simulation CSVs were found for the comparison.")
  }
  manifest <- do.call(rbind, rows)
  if (missing > 0L) {
    warning(sprintf("Missing %d requested simulation CSV combination(s).", missing))
  }
  manifest
}

message(sprintf("Using %s revisit simulation CSVs from %s", source_arg, input_dir))
message(sprintf("Minimum samples per dot: %d", minimum_samples))
manifest <- build_file_manifest()
message(sprintf("Found %d simulation CSV file(s).", nrow(manifest)))

process_simulation_file <- function(path, meta) {
  cols <- header_cols(path)
  selected_cols <- unique(c(
    intersect(c("graph", "node", "actual_reward", "V", "chosen_path"), cols),
    grep("^(expanded_node_t|expanded_reward_t|stop_t|kl_d_t)[0-9]+$", cols, value = TRUE),
    grep("^(terminal_choice_prob|action_output)_path[0-9]+_t[0-9]+$", cols, value = TRUE)
  ))
  dat <- read_csv_fast(path, select = selected_cols)
  if (nrow(dat) == 0L) {
    return(list(average = data.frame(), first_kl = data.frame(), pre_stop = data.frame(), stop_kl = data.frame()))
  }
  trial_cols <- setdiff(names(dat), c("node", "actual_reward"))
  if ("graph" %in% names(dat)) {
    trial_data <- dat[!duplicated(dat$graph), trial_cols, drop = FALSE]
  } else {
    trial_data <- dat[, trial_cols, drop = FALSE]
  }
  n_trials <- nrow(trial_data)
  if (n_trials == 0L) {
    return(list(average = data.frame(), first_kl = data.frame(), pre_stop = data.frame(), stop_kl = data.frame()))
  }
  trial_data$family <- meta$family
  trial_data$parameter_value <- meta$parameter_value
  trial_data$parameter_label <- meta$parameter_label
  trial_data$beta <- meta$beta
  trial_data$opportunity <- meta$opportunity
  trial_data$seed <- meta$seed
  trial_data$sigma <- meta$sigma

  reward_timesteps <- column_timesteps(names(trial_data), "expanded_reward_t")
  node_timesteps <- column_timesteps(names(trial_data), "expanded_node_t")
  kl_timesteps <- column_timesteps(names(trial_data), "kl_d_t")
  stop_timesteps <- column_timesteps(names(trial_data), "stop_t")
  terminal_timesteps <- sort(unique(suppressWarnings(as.integer(sub("^.*_t", "", grep(
    "^(terminal_choice_prob|action_output)_path[0-9]+_t[0-9]+$",
    names(trial_data),
    value = TRUE
  ))))))
  terminal_timesteps <- terminal_timesteps[is.finite(terminal_timesteps)]
  observation_timesteps <- sort(unique(intersect(reward_timesteps, node_timesteps)))

  reward_cols <- paste0("expanded_reward_t", reward_timesteps)
  reward_cols <- reward_cols[reward_cols %in% names(trial_data)]
  node_cols <- paste0("expanded_node_t", node_timesteps)
  node_cols <- node_cols[node_cols %in% names(trial_data)]
  kl_cols <- paste0("kl_d_t", kl_timesteps)
  kl_cols <- kl_cols[kl_cols %in% names(trial_data)]
  stop_cols <- paste0("stop_t", stop_timesteps)
  stop_cols <- stop_cols[stop_cols %in% names(trial_data)]
  reward_mat <- numeric_matrix_from_cols(trial_data, reward_cols)
  node_mat <- numeric_matrix_from_cols(trial_data, node_cols)
  kl_mat <- numeric_matrix_from_cols(trial_data, kl_cols)
  kl_mat[!is.finite(kl_mat)] <- 0
  stop_mat <- logical_matrix_from_cols(trial_data, stop_cols)
  stop_mat[is.na(stop_mat)] <- FALSE

  actual_reward_mat <- node_actual_reward_matrix(dat, trial_data)
  path_reward_mat <- path_reward_matrix_from_node_rewards(actual_reward_mat)
  tie_rows <- actual_path_reward_tie_flags(path_reward_mat)

  max_decision_timestep <- max(c(reward_timesteps, node_timesteps, stop_timesteps, kl_timesteps, terminal_timesteps), na.rm = TRUE)
  if (!is.finite(max_decision_timestep)) {
    max_decision_timestep <- ncol(reward_mat)
  }
  has_stop <- if (ncol(stop_mat) > 0L) rowSums(stop_mat) > 0L else rep(FALSE, n_trials)
  first_stop_timestep <- rep(NA_real_, n_trials)
  if (ncol(stop_mat) > 0L && any(has_stop)) {
    first_stop_index <- max.col(stop_mat, ties.method = "first")
    first_stop_timestep[has_stop] <- stop_timesteps[first_stop_index[has_stop]]
  }
  stop_decision_timestep <- ifelse(is.finite(first_stop_timestep), first_stop_timestep, max_decision_timestep)
  timestep_before_stop <- pmax(stop_decision_timestep - 1, 0)
  observations_before_stop <- rowSums(is.finite(reward_mat))
  chosen_path_reward <- suppressWarnings(as.numeric(trial_data$V))
  reward_norm_denominator <- realized_reward_denominator(path_reward_mat)
  normalized_reward <- chosen_path_reward / reward_norm_denominator
  kl_paid_total <- rowSums(kl_mat)

  first_node_col <- first_finite_col(node_mat)
  first_node <- rep(NA_integer_, n_trials)
  has_first <- !is.na(first_node_col)
  if (any(has_first)) {
    first_node[has_first] <- suppressWarnings(as.integer(node_mat[cbind(which(has_first), first_node_col[has_first])]))
  }
  first_path <- path_id_for_node(first_node)
  first_path_reward <- rep(NA_real_, n_trials)
  valid_first_path <- !is.na(first_path) & first_path >= 1L & first_path <= ncol(path_reward_mat)
  if (any(valid_first_path)) {
    rows <- which(valid_first_path)
    first_path_reward[rows] <- path_reward_mat[cbind(rows, first_path[rows])]
  }
  path_reward_sums <- rowSums(path_reward_mat)
  mean_other_path <- rep(NA_real_, n_trials)
  rows <- which(valid_first_path)
  if (length(rows) > 0L && task_path_count > 1L) {
    mean_other_path[rows] <- (path_reward_sums[rows] - first_path_reward[rows]) / (task_path_count - 1L)
  }
  abs_first_minus_mean_other <- abs(first_path_reward - mean_other_path)

  first_reward_t1 <- rep(NA_real_, n_trials)
  if ("expanded_node_t1" %in% names(trial_data)) {
    fallback <- if ("expanded_reward_t1" %in% names(trial_data)) trial_data$expanded_reward_t1 else NULL
    first_reward_t1 <- actual_reward_for_observed_node(trial_data$expanded_node_t1, actual_reward_mat, fallback)
  }

  average <- data.frame(
    family = meta$family,
    parameter_value = meta$parameter_value,
    parameter_label = meta$parameter_label,
    beta = meta$beta,
    opportunity = meta$opportunity,
    sigma = meta$sigma,
    seed = meta$seed,
    timestep_before_stop = timestep_before_stop,
    normalized_chosen_path_reward = normalized_reward,
    kl_paid_total = kl_paid_total,
    stringsAsFactors = FALSE
  )
  average <- average[is.finite(average$normalized_chosen_path_reward), , drop = FALSE]

  first_kl <- data.frame()
  if ("kl_d_t2" %in% names(trial_data)) {
    kl_first <- suppressWarnings(as.numeric(trial_data$kl_d_t2))
    keep <- is.finite(timestep_before_stop) & timestep_before_stop > 1 &
      is.finite(first_reward_t1) & is.finite(kl_first)
    if (any(keep)) {
      first_kl <- data.frame(
        family = meta$family,
        parameter_value = meta$parameter_value,
        parameter_label = meta$parameter_label,
        beta = meta$beta,
        opportunity = meta$opportunity,
        sigma = meta$sigma,
        seed = meta$seed,
        first_observed_reward_t1 = first_reward_t1[keep],
        kl_paid_at_first_timestep_after_continue = kl_first[keep],
        stringsAsFactors = FALSE
      )
    }
  }

  pre_stop_rows <- list()
  candidate_timesteps <- sort(unique(c(observation_timesteps, kl_timesteps, terminal_timesteps)))
  candidate_timesteps <- candidate_timesteps[is.finite(candidate_timesteps) & candidate_timesteps > 0]
  for (timestep in candidate_timesteps) {
    before_or_at_stop <- is.finite(timestep_before_stop) & timestep_before_stop >= timestep
    strict_pre_stop <- is.finite(timestep_before_stop) & timestep_before_stop > timestep
    if (!any(before_or_at_stop) && !any(strict_pre_stop)) {
      next
    }
    kl_col <- paste0("kl_d_t", timestep + 1L)
    kl_values <- if (kl_col %in% names(trial_data)) suppressWarnings(as.numeric(trial_data[[kl_col]])) else rep(NA_real_, n_trials)
    kl_values <- ifelse(strict_pre_stop, kl_values, NA_real_)
    entropy_values <- terminal_binary_choice_entropy_for_timestep(trial_data, timestep, tie_rows)
    keep <- before_or_at_stop & (is.finite(kl_values) | (strict_pre_stop & is.finite(entropy_values)))
    if (!any(keep)) {
      next
    }
    pre_stop_rows[[length(pre_stop_rows) + 1L]] <- data.frame(
      family = meta$family,
      parameter_value = meta$parameter_value,
      parameter_label = meta$parameter_label,
      beta = meta$beta,
      opportunity = meta$opportunity,
      sigma = meta$sigma,
      seed = meta$seed,
      timestep = timestep,
      timestep_before_stop = timestep_before_stop[keep],
      strict_pre_stop_timestep = timestep,
      kl_paid_at_timestep = kl_values[keep],
      terminal_binary_choice_entropy_at_timestep = ifelse(strict_pre_stop[keep], entropy_values[keep], NA_real_),
      stringsAsFactors = FALSE
    )
  }
  pre_stop <- if (length(pre_stop_rows) == 0L) data.frame() else do.call(rbind, pre_stop_rows)

  kl_paid_at_stop_timestep <- rep(NA_real_, n_trials)
  stop_observation_timestep <- suppressWarnings(as.integer(round(timestep_before_stop)))
  stop_kl_col <- match(stop_observation_timestep, kl_timesteps)
  valid_stop_kl <- !is.na(stop_kl_col) & is.finite(timestep_before_stop) & timestep_before_stop > 0
  if (any(valid_stop_kl)) {
    stop_rows <- which(valid_stop_kl)
    kl_paid_at_stop_timestep[stop_rows] <- kl_mat[cbind(stop_rows, stop_kl_col[stop_rows])]
  }
  keep_stop_kl <- is.finite(abs_first_minus_mean_other) &
    is.finite(kl_paid_at_stop_timestep) &
    is.finite(observations_before_stop) &
    observations_before_stop > 1
  stop_kl <- if (any(keep_stop_kl)) {
    data.frame(
      family = meta$family,
      parameter_value = meta$parameter_value,
      parameter_label = meta$parameter_label,
      beta = meta$beta,
      opportunity = meta$opportunity,
      sigma = meta$sigma,
      seed = meta$seed,
      absolute_first_observed_minus_mean_other_path = abs_first_minus_mean_other[keep_stop_kl],
      kl_paid_at_stop_timestep_after_first_continue = kl_paid_at_stop_timestep[keep_stop_kl],
      stringsAsFactors = FALSE
    )
  } else {
    data.frame()
  }

  list(average = average, first_kl = first_kl, pre_stop = pre_stop, stop_kl = stop_kl)
}

combine_data_frames <- function(pieces, name) {
  pieces <- lapply(pieces, `[[`, name)
  pieces <- pieces[vapply(pieces, nrow, integer(1)) > 0L]
  if (length(pieces) == 0L) {
    return(data.frame())
  }
  if (requireNamespace("data.table", quietly = TRUE)) {
    return(as.data.frame(data.table::rbindlist(pieces, fill = TRUE)))
  }
  do.call(rbind, pieces)
}

processed <- vector("list", nrow(manifest))
for (i in seq_len(nrow(manifest))) {
  processed[[i]] <- process_simulation_file(manifest$file[[i]], manifest[i, , drop = FALSE])
  if (i %% 25L == 0L || i == nrow(manifest)) {
    message(sprintf("Processed %d/%d CSVs", i, nrow(manifest)))
  }
}

average_data <- combine_data_frames(processed, "average")
first_kl_data <- combine_data_frames(processed, "first_kl")
pre_stop_data <- combine_data_frames(processed, "pre_stop")
stop_kl_data <- combine_data_frames(processed, "stop_kl")

mean_or_na <- function(x) {
  x <- suppressWarnings(as.numeric(x))
  x <- x[is.finite(x)]
  if (length(x) == 0L) NA_real_ else mean(x)
}

sem_or_na <- function(x) {
  x <- suppressWarnings(as.numeric(x))
  x <- x[is.finite(x)]
  if (length(x) < 2L) NA_real_ else stats::sd(x) / sqrt(length(x))
}

summarize_metric <- function(dat, group_cols, value_col, count_col = "n") {
  if (nrow(dat) == 0L || !value_col %in% names(dat)) {
    return(data.frame())
  }
  keep <- is.finite(suppressWarnings(as.numeric(dat[[value_col]])))
  dat <- dat[keep, , drop = FALSE]
  if (nrow(dat) == 0L) {
    return(data.frame())
  }
  group_cols <- group_cols[group_cols %in% names(dat)]
  seed_group_cols <- unique(c(group_cols, "seed"))
  if (requireNamespace("data.table", quietly = TRUE)) {
    dt <- data.table::as.data.table(dat)
    out <- dt[, .(
      value = mean_or_na(get(value_col)),
      n = .N
    ), by = group_cols]
    seed_means <- dt[, .(value = mean_or_na(get(value_col))), by = seed_group_cols]
    seed_sem <- seed_means[, .(seed_sem = sem_or_na(value)), by = group_cols]
    out <- merge(out, seed_sem, by = group_cols, all.x = TRUE)
    out <- as.data.frame(out)
  } else {
    out <- aggregate(dat[[value_col]], by = dat[, group_cols, drop = FALSE], FUN = mean_or_na)
    names(out)[names(out) == "x"] <- "value"
    counts <- aggregate(rep(1, nrow(dat)), by = dat[, group_cols, drop = FALSE], FUN = sum)
    names(counts)[names(counts) == "x"] <- "n"
    out <- merge(out, counts, by = group_cols, all.x = TRUE)
    seed_means <- aggregate(dat[[value_col]], by = dat[, seed_group_cols, drop = FALSE], FUN = mean_or_na)
    names(seed_means)[names(seed_means) == "x"] <- "value"
    seed_sem <- aggregate(seed_means$value, by = seed_means[, group_cols, drop = FALSE], FUN = sem_or_na)
    names(seed_sem)[names(seed_sem) == "x"] <- "seed_sem"
    out <- merge(out, seed_sem, by = group_cols, all.x = TRUE)
  }
  names(out)[names(out) == "value"] <- value_col
  names(out)[names(out) == "seed_sem"] <- paste0(value_col, "_seed_sem")
  names(out)[names(out) == "n"] <- count_col
  out <- out[out[[count_col]] >= minimum_samples, , drop = FALSE]
  out
}

first_kl_summary <- summarize_metric(
  first_kl_data,
  c("family", "parameter_value", "parameter_label", "sigma", "first_observed_reward_t1"),
  "kl_paid_at_first_timestep_after_continue"
)
kl_timestep_summary <- summarize_metric(
  pre_stop_data,
  c("family", "parameter_value", "parameter_label", "sigma", "timestep"),
  "kl_paid_at_timestep"
)
kl_timestep_by_total_summary <- summarize_metric(
  pre_stop_data,
  c("family", "parameter_value", "parameter_label", "sigma", "timestep_before_stop", "timestep"),
  "kl_paid_at_timestep"
)
entropy_timestep_summary <- summarize_metric(
  pre_stop_data,
  c("family", "parameter_value", "parameter_label", "sigma", "strict_pre_stop_timestep"),
  "terminal_binary_choice_entropy_at_timestep"
)
stop_kl_summary <- summarize_metric(
  stop_kl_data,
  c("family", "parameter_value", "parameter_label", "sigma", "absolute_first_observed_minus_mean_other_path"),
  "kl_paid_at_stop_timestep_after_first_continue"
)
reward_time_summary <- summarize_metric(
  average_data,
  c("family", "parameter_value", "parameter_label", "sigma"),
  "timestep_before_stop"
)
reward_time_x_summary <- summarize_metric(
  average_data,
  c("family", "parameter_value", "parameter_label", "sigma"),
  "normalized_chosen_path_reward"
)
if (nrow(reward_time_summary) > 0L && nrow(reward_time_x_summary) > 0L) {
  reward_time_summary <- merge(
    reward_time_summary,
    reward_time_x_summary[, c("family", "parameter_value", "parameter_label", "sigma", "normalized_chosen_path_reward"), drop = FALSE],
    by = c("family", "parameter_value", "parameter_label", "sigma"),
    all.x = TRUE
  )
}
reward_kl_summary <- summarize_metric(
  average_data,
  c("family", "parameter_value", "parameter_label", "sigma"),
  "kl_paid_total"
)
reward_kl_x_summary <- summarize_metric(
  average_data,
  c("family", "parameter_value", "parameter_label", "sigma"),
  "normalized_chosen_path_reward"
)
if (nrow(reward_kl_summary) > 0L && nrow(reward_kl_x_summary) > 0L) {
  reward_kl_summary <- merge(
    reward_kl_summary,
    reward_kl_x_summary[, c("family", "parameter_value", "parameter_label", "sigma", "normalized_chosen_path_reward"), drop = FALSE],
    by = c("family", "parameter_value", "parameter_label", "sigma"),
    all.x = TRUE
  )
}

first_kl_sigma_summary <- summarize_metric(
  first_kl_data,
  c("family", "parameter_value", "parameter_label", "sigma"),
  "kl_paid_at_first_timestep_after_continue"
)
kl_timestep_sigma_summary <- summarize_metric(
  pre_stop_data,
  c("family", "parameter_value", "parameter_label", "sigma"),
  "kl_paid_at_timestep"
)
pre_stop_data_after_first_paid_kl <- pre_stop_data
if (nrow(pre_stop_data_after_first_paid_kl) > 0L && "timestep" %in% names(pre_stop_data_after_first_paid_kl)) {
  pre_stop_data_after_first_paid_kl$timestep <- suppressWarnings(as.numeric(pre_stop_data_after_first_paid_kl$timestep))
  pre_stop_data_after_first_paid_kl <- pre_stop_data_after_first_paid_kl[
    is.finite(pre_stop_data_after_first_paid_kl$timestep) &
      pre_stop_data_after_first_paid_kl$timestep > 1,
    ,
    drop = FALSE
  ]
}
kl_timestep_except_first_sigma_summary <- summarize_metric(
  pre_stop_data_after_first_paid_kl,
  c("family", "parameter_value", "parameter_label", "sigma"),
  "kl_paid_at_timestep"
)
stop_kl_sigma_summary <- summarize_metric(
  stop_kl_data,
  c("family", "parameter_value", "parameter_label", "sigma"),
  "kl_paid_at_stop_timestep_after_first_continue"
)

run_folder <- sprintf(
  "%s_beta_vs_opportunity%s_%s",
  tree_label,
  if (identical(sampled_lambda_critic, "value")) "_vcritic" else "",
  format(Sys.time(), "%Y%m%d_%H%M%S")
)
plot_output_dir <- file.path(output_root, "revisit_compare", run_folder)
dir.create(plot_output_dir, recursive = TRUE, showWarnings = FALSE)
message(sprintf("Saving comparison plots to %s", plot_output_dir))

plot_font_size_pt <- 7
panel_plot_width_in <- 15 / 25.4
panel_plot_height_in <- 33 / 25.4
panel_margin_line_height_in <- plot_font_size_pt * 1.2 / 72
comparison_panel_margins <- c(bottom = 2.4, left = 3.2, top = 1.4, right = 0.8)
comparison_outer_margins <- c(bottom = 3.2, left = 3.2, top = 0.4, right = 0.2)
comparison_legend_width_in <- max(1.05, panel_plot_width_in * 0.9)
comparison_sigma_gap_width_in <- 4 / 25.4
comparison_y_label_line <- 1.15

comparison_panel_margins_for <- function(show_y_axis = FALSE) {
  if (isTRUE(show_y_axis)) {
    c(bottom = 2.2, left = 2.4, top = 1.2, right = 0.15)
  } else {
    c(bottom = 2.2, left = 0.35, top = 1.2, right = 0.15)
  }
}

comparison_panel_cell_width_in <- function(margins = comparison_panel_margins) {
  panel_plot_width_in + (margins[["left"]] + margins[["right"]]) * panel_margin_line_height_in
}

comparison_panel_cell_height_in <- function(margins = comparison_panel_margins) {
  panel_plot_height_in + (margins[["bottom"]] + margins[["top"]]) * panel_margin_line_height_in
}

open_comparison_png <- function(path, n_cols, n_rows, margins = comparison_panel_margins, layout_widths = NULL) {
  panel_width_total <- if (is.null(layout_widths)) {
    comparison_panel_cell_width_in(margins) * n_cols
  } else {
    sum(layout_widths)
  }
  grDevices::png(
    path,
    width = panel_width_total +
      comparison_legend_width_in +
      (comparison_outer_margins[["left"]] + comparison_outer_margins[["right"]]) * panel_margin_line_height_in,
    height = comparison_panel_cell_height_in(margins) * n_rows +
      (comparison_outer_margins[["bottom"]] + comparison_outer_margins[["top"]]) * panel_margin_line_height_in,
    units = "in",
    res = 300,
    pointsize = plot_font_size_pt
  )
}

panel_center_adj <- function(panel_widths, legend_width = comparison_legend_width_in) {
  panel_total <- sum(panel_widths)
  panel_total / (panel_total + legend_width) / 2
}

format_ticks <- function(x) vapply(x, num_label, character(1))

family_title <- function(family) {
  if (identical(family, "beta")) "Beta varies (opportunity = 0)" else "Opportunity varies (beta = 1000)"
}

family_color_values <- function(family, params) {
  params <- sort(unique(suppressWarnings(as.numeric(params))))
  if (length(params) == 0L) {
    return(character())
  }
  palette <- if (identical(family, "beta")) {
    # Lower beta should be darker.
    grDevices::colorRampPalette(c("#00441b", "#238b45", "#74c476"))
  } else {
    # Higher VAE opportunity cost should be darker.
    grDevices::colorRampPalette(c("#6baed6", "#2171b5", "#08306b"))
  }
  cols <- palette(max(length(params), 2L))[seq_along(params)]
  names(cols) <- as.character(params)
  cols
}

series_color <- function(family, parameter_value) {
  vals <- if (identical(family, "beta")) beta_values else opportunity_values
  colors <- family_color_values(family, as_num(vals))
  colors[[as.character(suppressWarnings(as.numeric(parameter_value)))]]
}

series_pch <- function(family) if (identical(family, "beta")) 16 else 17

safe_ylim <- function(values, sem_values = NULL) {
  values <- suppressWarnings(as.numeric(values))
  sem_values <- suppressWarnings(as.numeric(sem_values))
  candidates <- c(values, values - sem_values, values + sem_values)
  candidates <- candidates[is.finite(candidates)]
  if (length(candidates) == 0L) {
    return(c(0, 1))
  }
  lim <- range(candidates)
  if (abs(diff(lim)) < 1e-12) {
    lim <- lim + c(-0.5, 0.5)
  }
  pad <- diff(lim) * 0.08
  lim + c(-pad, pad)
}

safe_xlim <- function(values, start_at_one = FALSE) {
  values <- suppressWarnings(as.numeric(values))
  values <- values[is.finite(values)]
  if (length(values) == 0L) {
    return(if (isTRUE(start_at_one)) c(1, 2) else c(0, 1))
  }
  lim <- range(values)
  if (abs(diff(lim)) < 1e-12) {
    lim <- lim + c(-0.5, 0.5)
  }
  pad <- diff(lim) * 0.04
  lim <- lim + c(-pad, pad)
  if (isTRUE(start_at_one)) {
    lim[1L] <- max(1, lim[1L])
  }
  lim
}

is_timestep_x_col <- function(x_col) {
  x_col %in% c("timestep", "strict_pre_stop_timestep", "timestep_before_stop")
}

draw_timestep_x_axis <- function(values) {
  values <- suppressWarnings(as.numeric(values))
  values <- values[is.finite(values)]
  if (length(values) == 0L) {
    axis(1, at = 1)
    return(invisible(NULL))
  }
  max_tick <- max(1L, ceiling(max(values)))
  axis(1, at = seq.int(1L, max_tick))
  invisible(NULL)
}

draw_error_bars <- function(x, y, sem, col) {
  sem <- suppressWarnings(as.numeric(sem))
  keep <- is.finite(x) & is.finite(y) & is.finite(sem) & sem > 0
  if (any(keep)) {
    graphics::arrows(x[keep], y[keep] - sem[keep], x[keep], y[keep] + sem[keep],
      angle = 90, code = 3, length = 0.025, col = col
    )
  }
}

plot_sigma_panel_lines <- function(summary_data, x_col, y_col, file_name, xlab, ylab) {
  if (nrow(summary_data) == 0L || !all(c(x_col, y_col) %in% names(summary_data))) {
    warning(sprintf("No data for %s; skipping.", file_name))
    return(invisible(NULL))
  }
  sem_col <- paste0(y_col, "_seed_sem")
  sigma_levels <- sort(unique(suppressWarnings(as.numeric(summary_data$sigma))))
  sigma_levels <- sigma_levels[is.finite(sigma_levels)]
  family_levels <- c("beta", "opportunity")
  n_plot_panels <- length(family_levels) * length(sigma_levels)
  n_rows <- 1L
  path <- file.path(plot_output_dir, file_name)
  panel_layout <- integer()
  panel_widths <- numeric()
  panel_id <- 0L
  for (sigma_i in seq_along(sigma_levels)) {
    for (family_i in seq_along(family_levels)) {
      panel_id <- panel_id + 1L
      show_y_axis <- panel_id == 1L
      panel_layout <- c(panel_layout, panel_id)
      panel_widths <- c(panel_widths, comparison_panel_cell_width_in(comparison_panel_margins_for(show_y_axis)))
    }
    if (sigma_i < length(sigma_levels)) {
      panel_layout <- c(panel_layout, 0L)
      panel_widths <- c(panel_widths, comparison_sigma_gap_width_in)
    }
  }
  open_comparison_png(path, n_cols = n_plot_panels, n_rows = n_rows, layout_widths = panel_widths)
  layout(matrix(c(panel_layout, n_plot_panels + 1L), nrow = n_rows),
    widths = c(panel_widths, comparison_legend_width_in)
  )
  old_par <- par(no.readonly = TRUE)
  par(cex = 1, cex.axis = 1, cex.lab = 1, cex.main = 1, oma = comparison_outer_margins)
  y_lim <- safe_ylim(summary_data[[y_col]], if (sem_col %in% names(summary_data)) summary_data[[sem_col]] else NULL)
  x_is_timestep <- is_timestep_x_col(x_col)
  x_lim <- safe_xlim(summary_data[[x_col]], start_at_one = x_is_timestep)
  panel_i <- 0L
  for (sigma_value in sigma_levels) {
    for (family in family_levels) {
      panel_i <- panel_i + 1L
      panel_data <- summary_data[
        summary_data$family == family &
          parameter_equal(summary_data$sigma, sigma_value),
        ,
        drop = FALSE
      ]
      show_y_axis <- panel_i == 1L
      par(mar = comparison_panel_margins_for(show_y_axis = show_y_axis))
      plot(NA,
        xlim = x_lim,
        ylim = y_lim,
        xlab = "",
        ylab = "",
        main = if (identical(family, family_levels[[1L]])) sprintf("sigma = %s", num_label(sigma_value)) else "",
        xaxt = if (x_is_timestep) "n" else "s",
        yaxt = if (show_y_axis) "s" else "n",
        cex.lab = 1,
        cex.axis = 1,
        cex.main = 1
      )
      if (x_is_timestep) {
        draw_timestep_x_axis(summary_data[[x_col]])
      }
      grid(col = "grey90")
      params <- sort(unique(suppressWarnings(as.numeric(panel_data$parameter_value))))
      for (param in params) {
        line_data <- panel_data[parameter_equal(panel_data$parameter_value, param), , drop = FALSE]
        line_data <- line_data[order(suppressWarnings(as.numeric(line_data[[x_col]]))), , drop = FALSE]
        if (nrow(line_data) == 0L) {
          next
        }
        x <- suppressWarnings(as.numeric(line_data[[x_col]]))
        y <- suppressWarnings(as.numeric(line_data[[y_col]]))
        col <- series_color(family, param)
        pch <- series_pch(family)
        lines(x, y, col = col, lwd = 1.3)
        points(x, y, col = col, pch = pch, cex = 0.7)
        if (sem_col %in% names(line_data)) {
          draw_error_bars(x, y, suppressWarnings(as.numeric(line_data[[sem_col]])), col)
        }
      }
    }
  }
  par(mar = c(4.6, 0.2, 3.1, 0.2))
  plot.new()
  legend_items <- list()
  for (family in family_levels) {
    params <- if (identical(family, "beta")) as_num(beta_values) else as_num(opportunity_values)
    params <- sort(unique(params[is.finite(params)]))
    for (param in params) {
      legend_items[[length(legend_items) + 1L]] <- list(
        label = if (identical(family, "beta")) paste0("beta ", num_label(param)) else paste0("opp ", num_label(param)),
        col = series_color(family, param),
        pch = series_pch(family)
      )
    }
  }
  if (length(legend_items) > 0L) {
    graphics::legend(
      "center",
      legend = vapply(legend_items, `[[`, character(1), "label"),
      col = vapply(legend_items, `[[`, character(1), "col"),
      pch = vapply(legend_items, `[[`, numeric(1), "pch"),
      lwd = 1.3,
      bty = "n",
      cex = 1
    )
  }
  mtext(xlab, side = 1, outer = TRUE, line = 1.6, cex = 1, adj = panel_center_adj(panel_widths))
  mtext(ylab, side = 2, outer = TRUE, line = comparison_y_label_line, cex = 1)
  invisible(try(par(old_par), silent = TRUE))
  grDevices::dev.off()
  message(sprintf("Saved %s", path))
  invisible(path)
}

plot_sigma_panel_lines_by_total_timestep <- function(summary_data, x_col, y_col, file_name, xlab, ylab) {
  required <- c(x_col, y_col, "timestep_before_stop")
  if (nrow(summary_data) == 0L || !all(required %in% names(summary_data))) {
    warning(sprintf("No data for %s; skipping.", file_name))
    return(invisible(NULL))
  }
  summary_data <- summary_data[
    is.finite(suppressWarnings(as.numeric(summary_data$timestep_before_stop))),
    ,
    drop = FALSE
  ]
  if (nrow(summary_data) == 0L) {
    warning(sprintf("No finite total-timestep data for %s; skipping.", file_name))
    return(invisible(NULL))
  }
  sem_col <- paste0(y_col, "_seed_sem")
  sigma_levels <- sort(unique(suppressWarnings(as.numeric(summary_data$sigma))))
  sigma_levels <- sigma_levels[is.finite(sigma_levels)]
  total_levels <- sort(unique(suppressWarnings(as.numeric(summary_data$timestep_before_stop))))
  total_levels <- total_levels[is.finite(total_levels)]
  family_levels <- c("beta", "opportunity")
  if (length(sigma_levels) == 0L || length(total_levels) == 0L) {
    warning(sprintf("No finite sigma/total-timestep levels for %s; skipping.", file_name))
    return(invisible(NULL))
  }
  n_plot_panels <- length(family_levels) * length(sigma_levels)
  n_rows <- length(total_levels)
  path <- file.path(plot_output_dir, file_name)
  panel_widths <- numeric()
  for (sigma_i in seq_along(sigma_levels)) {
    for (family_i in seq_along(family_levels)) {
      show_y_axis <- sigma_i == 1L && family_i == 1L
      panel_widths <- c(panel_widths, comparison_panel_cell_width_in(comparison_panel_margins_for(show_y_axis)))
    }
    if (sigma_i < length(sigma_levels)) {
      panel_widths <- c(panel_widths, comparison_sigma_gap_width_in)
    }
  }
  row_layouts <- vector("list", n_rows)
  panel_id <- 0L
  for (row_i in seq_len(n_rows)) {
    row_layout <- integer()
    for (sigma_i in seq_along(sigma_levels)) {
      for (family_i_unused in seq_along(family_levels)) {
        panel_id <- panel_id + 1L
        row_layout <- c(row_layout, panel_id)
      }
      if (sigma_i < length(sigma_levels)) {
        row_layout <- c(row_layout, 0L)
      }
    }
    row_layouts[[row_i]] <- row_layout
  }
  panel_matrix <- do.call(rbind, row_layouts)
  legend_id <- panel_id + 1L
  open_comparison_png(path, n_cols = n_plot_panels, n_rows = n_rows, layout_widths = panel_widths)
  layout(cbind(panel_matrix, rep(legend_id, n_rows)),
    widths = c(panel_widths, comparison_legend_width_in)
  )
  old_par <- par(no.readonly = TRUE)
  par(cex = 1, cex.axis = 1, cex.lab = 1, cex.main = 1, oma = comparison_outer_margins)
  y_lim <- safe_ylim(summary_data[[y_col]], if (sem_col %in% names(summary_data)) summary_data[[sem_col]] else NULL)
  x_is_timestep <- is_timestep_x_col(x_col)
  x_lim <- safe_xlim(summary_data[[x_col]], start_at_one = x_is_timestep)
  for (row_i in seq_along(total_levels)) {
    total_value <- total_levels[[row_i]]
    row_data <- summary_data[
      parameter_equal(summary_data$timestep_before_stop, total_value),
      ,
      drop = FALSE
    ]
    panel_i <- 0L
    for (sigma_i in seq_along(sigma_levels)) {
      sigma_value <- sigma_levels[[sigma_i]]
      for (family_i in seq_along(family_levels)) {
        family <- family_levels[[family_i]]
        panel_i <- panel_i + 1L
        panel_data <- row_data[
          row_data$family == family &
            parameter_equal(row_data$sigma, sigma_value),
          ,
          drop = FALSE
        ]
        show_y_axis <- sigma_i == 1L && family_i == 1L
        par(mar = comparison_panel_margins_for(show_y_axis = show_y_axis))
        title_parts <- c()
        if (row_i == 1L && identical(family, family_levels[[1L]])) {
          title_parts <- c(title_parts, sprintf("sigma = %s", num_label(sigma_value)))
        }
        if (family_i == 1L) {
          title_parts <- c(title_parts, sprintf("total t = %s", num_label(total_value)))
        }
        plot(NA,
          xlim = x_lim,
          ylim = y_lim,
          xlab = "",
          ylab = "",
          main = paste(title_parts, collapse = "\n"),
          xaxt = if (x_is_timestep) "n" else "s",
          yaxt = if (show_y_axis) "s" else "n",
          cex.lab = 1,
          cex.axis = 1,
          cex.main = 1
        )
        if (x_is_timestep) {
          draw_timestep_x_axis(summary_data[[x_col]])
        }
        grid(col = "grey90")
        params <- sort(unique(suppressWarnings(as.numeric(panel_data$parameter_value))))
        for (param in params) {
          line_data <- panel_data[parameter_equal(panel_data$parameter_value, param), , drop = FALSE]
          line_data <- line_data[order(suppressWarnings(as.numeric(line_data[[x_col]]))), , drop = FALSE]
          if (nrow(line_data) == 0L) {
            next
          }
          x <- suppressWarnings(as.numeric(line_data[[x_col]]))
          y <- suppressWarnings(as.numeric(line_data[[y_col]]))
          col <- series_color(family, param)
          pch <- series_pch(family)
          lines(x, y, col = col, lwd = 1.2)
          points(x, y, col = col, pch = pch, cex = 0.65)
          if (sem_col %in% names(line_data)) {
            draw_error_bars(x, y, suppressWarnings(as.numeric(line_data[[sem_col]])), col)
          }
        }
      }
    }
  }
  par(mar = c(4.6, 0.2, 3.1, 0.2))
  plot.new()
  legend_items <- list()
  for (family in family_levels) {
    params <- if (identical(family, "beta")) as_num(beta_values) else as_num(opportunity_values)
    params <- sort(unique(params[is.finite(params)]))
    for (param in params) {
      legend_items[[length(legend_items) + 1L]] <- list(
        label = if (identical(family, "beta")) paste0("beta ", num_label(param)) else paste0("opp ", num_label(param)),
        col = series_color(family, param),
        pch = series_pch(family)
      )
    }
  }
  if (length(legend_items) > 0L) {
    graphics::legend(
      "center",
      legend = vapply(legend_items, `[[`, character(1), "label"),
      col = vapply(legend_items, `[[`, character(1), "col"),
      pch = vapply(legend_items, `[[`, numeric(1), "pch"),
      lwd = 1.3,
      bty = "n",
      cex = 1
    )
  }
  mtext(xlab, side = 1, outer = TRUE, line = 1.6, cex = 1, adj = panel_center_adj(panel_widths))
  mtext(ylab, side = 2, outer = TRUE, line = comparison_y_label_line, cex = 1)
  invisible(try(par(old_par), silent = TRUE))
  grDevices::dev.off()
  message(sprintf("Saved %s", path))
  invisible(path)
}

plot_sigma_summary <- function(summary_data, y_col, file_name, ylab) {
  if (nrow(summary_data) == 0L || !y_col %in% names(summary_data)) {
    warning(sprintf("No data for %s; skipping.", file_name))
    return(invisible(NULL))
  }
  sem_col <- paste0(y_col, "_seed_sem")
  family_levels <- c("beta", "opportunity")
  path <- file.path(plot_output_dir, file_name)
  panel_widths <- vapply(seq_along(family_levels), function(family_i) {
    comparison_panel_cell_width_in(comparison_panel_margins_for(show_y_axis = family_i == 1L))
  }, numeric(1))
  open_comparison_png(path, n_cols = length(family_levels), n_rows = 1L, layout_widths = panel_widths)
  layout(cbind(matrix(seq_along(family_levels), nrow = 1L), length(family_levels) + 1L),
    widths = c(panel_widths, comparison_legend_width_in)
  )
  old_par <- par(no.readonly = TRUE)
  par(cex = 1, cex.axis = 1, cex.lab = 1, cex.main = 1, oma = comparison_outer_margins)
  y_lim <- safe_ylim(summary_data[[y_col]], if (sem_col %in% names(summary_data)) summary_data[[sem_col]] else NULL)
  x_lim <- safe_xlim(summary_data$sigma)
  for (family_i in seq_along(family_levels)) {
    family <- family_levels[[family_i]]
    panel_data <- summary_data[summary_data$family == family, , drop = FALSE]
    show_y_axis <- family_i == 1L
    par(mar = comparison_panel_margins_for(show_y_axis = show_y_axis))
    plot(NA,
      xlim = x_lim,
      ylim = y_lim,
      xlab = "",
      ylab = "",
      main = "",
      yaxt = if (show_y_axis) "s" else "n",
      cex.lab = 1,
      cex.axis = 1,
      cex.main = 1
    )
    grid(col = "grey90")
    params <- sort(unique(suppressWarnings(as.numeric(panel_data$parameter_value))))
    for (param in params) {
      line_data <- panel_data[parameter_equal(panel_data$parameter_value, param), , drop = FALSE]
      line_data <- line_data[order(suppressWarnings(as.numeric(line_data$sigma))), , drop = FALSE]
      x <- suppressWarnings(as.numeric(line_data$sigma))
      y <- suppressWarnings(as.numeric(line_data[[y_col]]))
      col <- series_color(family, param)
      pch <- series_pch(family)
      lines(x, y, col = col, lwd = 1.3)
      points(x, y, col = col, pch = pch, cex = 0.75)
      if (sem_col %in% names(line_data)) {
        draw_error_bars(x, y, suppressWarnings(as.numeric(line_data[[sem_col]])), col)
      }
    }
  }
  par(mar = c(4.6, 0.2, 3.1, 0.2))
  plot.new()
  legend_items <- list()
  for (family in family_levels) {
    params <- if (identical(family, "beta")) as_num(beta_values) else as_num(opportunity_values)
    params <- sort(unique(params[is.finite(params)]))
    for (param in params) {
      legend_items[[length(legend_items) + 1L]] <- list(
        label = if (identical(family, "beta")) paste0("beta ", num_label(param)) else paste0("opp ", num_label(param)),
        col = series_color(family, param),
        pch = series_pch(family)
      )
    }
  }
  if (length(legend_items) > 0L) {
    graphics::legend(
      "center",
      legend = vapply(legend_items, `[[`, character(1), "label"),
      col = vapply(legend_items, `[[`, character(1), "col"),
      pch = vapply(legend_items, `[[`, numeric(1), "pch"),
      lwd = 1.3,
      bty = "n",
      cex = 1
    )
  }
  mtext("Observation noise sigma", side = 1, outer = TRUE, line = 1.6, cex = 1, adj = panel_center_adj(panel_widths))
  mtext(ylab, side = 2, outer = TRUE, line = comparison_y_label_line, cex = 1)
  invisible(try(par(old_par), silent = TRUE))
  grDevices::dev.off()
  message(sprintf("Saved %s", path))
  invisible(path)
}

plot_sigma_panel_lines(
  first_kl_summary,
  "first_observed_reward_t1",
  "kl_paid_at_first_timestep_after_continue",
  "kl_paid_at_first_timestep_after_continue_vs_first_observed_reward_t1_sigma_panels.png",
  "Actual reward\nof first observed path\nat t1",
  "KL paid\nafter continuing from t1"
)

plot_sigma_panel_lines(
  stop_kl_summary,
  "absolute_first_observed_minus_mean_other_path",
  "kl_paid_at_stop_timestep_after_first_continue",
  "kl_paid_at_stop_timestep_after_first_continue_vs_absolute_first_observed_minus_mean_other_path_sigma_panels.png",
  "|first observed path\n- mean other path|",
  "KL paid at stop timestep\nafter first continue"
)

plot_sigma_panel_lines(
  kl_timestep_summary,
  "timestep",
  "kl_paid_at_timestep",
  "kl_paid_at_timestep_vs_timestep_sigma_panels.png",
  "Observation timestep",
  "KL paid at timestep"
)

plot_sigma_panel_lines_by_total_timestep(
  kl_timestep_by_total_summary,
  "timestep",
  "kl_paid_at_timestep",
  "kl_paid_at_timestep_vs_timestep_by_total_timestep_sigma_panels.png",
  "Observation timestep",
  "KL paid at timestep"
)

plot_sigma_panel_lines(
  reward_kl_summary,
  "normalized_chosen_path_reward",
  "kl_paid_total",
  "kl_paid_total_vs_normalized_chosen_path_reward_sigma_panels.png",
  "Normalized\nchosen path reward",
  "Total KL paid"
)

plot_sigma_panel_lines(
  entropy_timestep_summary,
  "strict_pre_stop_timestep",
  "terminal_binary_choice_entropy_at_timestep",
  "terminal_binary_choice_entropy_at_timestep_vs_strict_pre_stop_timestep_sigma_panels.png",
  "Strict pre-stop observation timestep",
  "Binary terminal-choice\nentropy"
)

plot_sigma_panel_lines(
  reward_time_summary,
  "normalized_chosen_path_reward",
  "timestep_before_stop",
  "timestep_before_stop_vs_normalized_chosen_path_reward_sigma_panels.png",
  "Normalized\nchosen path reward",
  "Observations before stop"
)

plot_sigma_summary(
  first_kl_sigma_summary,
  "kl_paid_at_first_timestep_after_continue",
  "summary_kl_paid_at_first_timestep_after_continue_vs_sigma.png",
  "KL paid after continuing from t1"
)

plot_sigma_summary(
  kl_timestep_sigma_summary,
  "kl_paid_at_timestep",
  "summary_kl_paid_at_timestep_vs_sigma.png",
  "KL paid at timestep"
)

plot_sigma_summary(
  kl_timestep_except_first_sigma_summary,
  "kl_paid_at_timestep",
  "summary_kl_paid_at_timestep_except_first_paid_kl_vs_sigma.png",
  "KL paid at timestep\nexcluding first paid KL"
)

plot_sigma_summary(
  stop_kl_sigma_summary,
  "kl_paid_at_stop_timestep_after_first_continue",
  "summary_kl_paid_at_stop_timestep_after_first_continue_vs_sigma.png",
  "KL paid at stop timestep\nafter first continue"
)

message("Done.")
