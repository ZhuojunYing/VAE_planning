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
file_suffix <- sprintf(
  "%s_lambda_%s_alpha_%s_beta_%s_opportunity_%s_expansion_%s_variant_%s_%s_%s_%s%s",
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
  source_suffix
)

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
  "^kl_d_t[0-9]+$"
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

numeric_file_match <- function(lambda_value, alpha_value, beta_value, opportunity_value, seed) {
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
        "_revisit_maxobs_([^_]+)_", input_type, "\\.csv$"
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
        if (all(abs(found - requested) < 1e-8) && abs(found_maxobs - requested_maxobs) < 1e-8) {
          return(files[[i]])
        }
      }
    }
  }
  NA_character_
}

simulation_path <- function(lambda_value, alpha_value, beta_value, opportunity_value, seed) {
  for (lambda_candidate in value_candidates(lambda_value)) {
    for (alpha_candidate in value_candidates(alpha_value)) {
      for (beta_candidate in value_candidates(beta_value)) {
        for (opportunity_candidate in value_candidates(opportunity_value)) {
          for (maxobs_candidate in value_candidates(max_observations_arg)) {
            for (tree_label_candidate in simulation_tree_file_labels()) {
              for (variant_file_segment in variant_file_segments) {
                file_name <- sprintf(
                  "lambda_%s_alpha_%s_beta_%s_opportunity_%s_expansion_%s_%sseed_%d_%s_revisit_maxobs_%s_%s.csv",
                  lambda_candidate,
                  alpha_candidate,
                  beta_candidate,
                  opportunity_candidate,
                  expansion_decision_version,
                  variant_file_segment,
                  seed,
                  tree_label_candidate,
                  maxobs_candidate,
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
  numeric_file_match(lambda_value, alpha_value, beta_value, opportunity_value, seed)
}

matching_simulation_files <- function(lambda_value, alpha_value, beta_value, opportunity_value) {
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
        "_revisit_maxobs_([^_]+)_", input_type, "\\.csv$"
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
        if (all(abs(found - requested) < 1e-8) && abs(found_maxobs - requested_maxobs) < 1e-8) {
          rows[[length(rows) + 1L]] <- data.frame(path = files[[i]], seed = seed_value, stringsAsFactors = FALSE)
        }
      }
    }
  }
  if (length(rows) == 0) {
    return(data.frame(path = character(), seed = integer(), stringsAsFactors = FALSE))
  }
  unique(do.call(rbind, rows))
}

read_seed_file <- function(beta_value, opportunity_value, seed) {
  file_path <- simulation_path(lambda_arg, alpha_arg, beta_value, opportunity_value, seed)
  if (is.na(file_path)) {
    warning(sprintf("Missing revisit simulation file for beta=%s opportunity=%s seed=%d", beta_value, opportunity_value, seed))
    return(NULL)
  }
  dat <- read_csv_fast(file_path, keep_names = simulation_keep_names, keep_patterns = simulation_keep_patterns)
  dat$beta <- beta_value
  dat$opportunity <- opportunity_value
  dat$seed <- seed
  dat$file_path <- file_path
  dat$source_file <- file_path
  dat
}

read_simulation_file <- function(file_path, beta_value, opportunity_value, seed) {
  dat <- read_csv_fast(file_path, keep_names = simulation_keep_names, keep_patterns = simulation_keep_patterns)
  dat$beta <- beta_value
  dat$opportunity <- opportunity_value
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
    if (auto_seeds) {
      combo_files <- matching_simulation_files(lambda_arg, alpha_arg, beta_value, opportunity_value)
      if (nrow(combo_files) == 0) {
        warning(sprintf("Missing revisit simulation files for beta=%s opportunity=%s", beta_value, opportunity_value))
      }
      for (file_i in seq_len(nrow(combo_files))) {
        loaded_data[[length(loaded_data) + 1L]] <- read_simulation_file(
          combo_files$path[[file_i]],
          beta_value,
          opportunity_value,
          combo_files$seed[[file_i]]
        )
      }
    } else {
      for (seed in seed_values) {
        seed_data <- read_seed_file(beta_value, opportunity_value, seed)
        if (!is.null(seed_data)) {
          loaded_data[[length(loaded_data) + 1L]] <- seed_data
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

unique_trial_rows <- function(dat, required_cols) {
  trial_id_cols <- intersect(c("beta", "opportunity", "seed", "graph"), names(dat))
  trial_cols <- unique(c(trial_id_cols, required_cols))
  unique(dat[, trial_cols, drop = FALSE])
}

trial_id_cols <- intersect(c("beta", "opportunity", "seed", "graph"), names(all_data))
reward_timesteps <- column_timesteps(all_data, "^expanded_reward_t[0-9]+$", "^expanded_reward_t")
node_timesteps <- column_timesteps(all_data, "^expanded_node_t[0-9]+$", "^expanded_node_t")
kl_timesteps <- column_timesteps(all_data, "^kl_d_t[0-9]+$", "^kl_d_t")
observation_timesteps <- Reduce(intersect, list(reward_timesteps, node_timesteps))

trial_required_cols <- unique(c(
  "V",
  paste0("expanded_node_t", node_timesteps),
  paste0("expanded_reward_t", reward_timesteps),
  paste0("kl_d_t", kl_timesteps)
))
trial_required_cols <- trial_required_cols[trial_required_cols %in% names(all_data)]
trial_data <- unique_trial_rows(all_data, trial_required_cols)

reward_cols <- paste0("expanded_reward_t", reward_timesteps)
reward_cols <- reward_cols[reward_cols %in% names(trial_data)]
node_cols <- paste0("expanded_node_t", node_timesteps)
node_cols <- node_cols[node_cols %in% names(trial_data)]
kl_cols <- paste0("kl_d_t", kl_timesteps)
kl_cols <- kl_cols[kl_cols %in% names(trial_data)]

reward_mat <- numeric_matrix_from_cols(trial_data, reward_cols)
node_mat <- numeric_matrix_from_cols(trial_data, node_cols)
kl_mat <- numeric_matrix_from_cols(trial_data, kl_cols)
kl_mat[!is.finite(kl_mat)] <- 0
trial_data$chosen_path_reward <- suppressWarnings(as.numeric(trial_data$V))
trial_data$normalized_chosen_path_reward <- trial_data$chosen_path_reward / task_reward_norm
trial_data$observations_before_stop <- rowSums(is.finite(reward_mat))
trial_data$unique_nodes_visited <- apply(node_mat, 1, function(row) {
  length(unique(row[is.finite(row)]))
})
trial_data$kl_paid_total <- rowSums(kl_mat)
trial_data <- trial_data[is.finite(trial_data$normalized_chosen_path_reward), , drop = FALSE]

mean_or_na <- function(x) {
  x <- suppressWarnings(as.numeric(x))
  x <- x[is.finite(x)]
  if (length(x) == 0) NA_real_ else mean(x)
}

aggregate_means_by <- function(dat, group_cols, value_cols) {
  pieces <- lapply(value_cols, function(value_col) {
    out <- aggregate(dat[[value_col]], by = dat[, group_cols, drop = FALSE], FUN = mean_or_na)
    names(out)[names(out) == "x"] <- value_col
    out
  })
  Reduce(function(left, right) merge(left, right, by = group_cols, all = TRUE), pieces)
}

average_summary <- aggregate_means_by(
  trial_data,
  group_cols = c("beta", "opportunity"),
  value_cols = c(
    "normalized_chosen_path_reward",
    "observations_before_stop",
    "unique_nodes_visited",
    "kl_paid_total"
  )
)
count_data <- aggregate(
  normalized_chosen_path_reward ~ beta + opportunity,
  data = trial_data,
  FUN = length
)
names(count_data)[names(count_data) == "normalized_chosen_path_reward"] <- "n"
average_summary <- merge(average_summary, count_data, by = c("beta", "opportunity"), all.x = TRUE)

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

plot_average_scatter <- function(summary_data, x_col, y_col, xlab, ylab, file_prefix) {
  path <- file.path(results_dir, sprintf("%s_%s.png", file_prefix, file_suffix))
  open_panel_png(path, n_cols = 1L, n_rows = 1L, legend_fraction = legend_panel_fraction)
  old_par <- par(no.readonly = TRUE)
  layout(matrix(c(1, 2), nrow = 1), widths = c(1, legend_panel_fraction))
  par(mar = c(4.2, 4.2, 1, 1))
  apply_panel_text_style()
  plot(
    NA,
    xlim = expand_range(summary_data[[x_col]], pad = 0.05),
    ylim = expand_range(summary_data[[y_col]], pad = 0.05),
    xlab = xlab,
    ylab = ylab,
    main = ""
  )
  grid()
  points(
    summary_data[[x_col]],
    summary_data[[y_col]],
    pch = 19,
    cex = 1.35,
    col = point_color_for(summary_data$beta, summary_data$opportunity, alpha = 0.6)
  )
  par(mar = c(0, 0, 0, 0))
  plot_parameter_legend()
  par(old_par)
  dev.off()
  message(sprintf("Saved %s", path))
}

plot_average_scatter(
  average_summary,
  x_col = "normalized_chosen_path_reward",
  y_col = "observations_before_stop",
  xlab = "Average normalized chosen path reward",
  ylab = "Average timestep before stopping",
  file_prefix = "revisit_average_timestep_before_stop_vs_average_normalized_chosen_path_reward"
)
plot_average_scatter(
  average_summary,
  x_col = "normalized_chosen_path_reward",
  y_col = "unique_nodes_visited",
  xlab = "Average normalized chosen path reward",
  ylab = "Average number of unique nodes visited",
  file_prefix = "revisit_average_unique_nodes_visited_vs_average_normalized_chosen_path_reward"
)
plot_average_scatter(
  average_summary,
  x_col = "normalized_chosen_path_reward",
  y_col = "kl_paid_total",
  xlab = "Average normalized chosen path reward",
  ylab = "Average KL paid across all timesteps",
  file_prefix = "revisit_average_kl_paid_vs_average_normalized_chosen_path_reward"
)
plot_average_scatter(
  average_summary,
  x_col = "observations_before_stop",
  y_col = "kl_paid_total",
  xlab = "Average timestep before stopping",
  ylab = "Average KL paid across all timesteps",
  file_prefix = "revisit_average_kl_paid_vs_average_timestep_before_stop"
)
plot_average_scatter(
  average_summary,
  x_col = "kl_paid_total",
  y_col = "unique_nodes_visited",
  xlab = "Average KL paid across all timesteps",
  ylab = "Average number of unique nodes visited",
  file_prefix = "revisit_average_unique_nodes_visited_vs_average_kl_paid"
)
plot_average_scatter(
  average_summary,
  x_col = "unique_nodes_visited",
  y_col = "observations_before_stop",
  xlab = "Average number of unique nodes visited",
  ylab = "Average timestep before stopping",
  file_prefix = "revisit_average_timestep_before_stop_vs_average_unique_nodes_visited"
)

build_node_visit_summary <- function(dat, trial_data, node_mat) {
  if (!all(c("node", "actual_reward") %in% names(dat))) {
    warning("Missing node/actual_reward columns; skipping node reward by visit-count plot.")
    return(data.frame())
  }
  id_cols <- intersect(c("beta", "opportunity", "seed", "graph"), names(dat))
  if (length(id_cols) == 0) {
    return(data.frame())
  }
  trial_ids <- unique(trial_data[, id_cols, drop = FALSE])
  visit_rows <- vector("list", task_node_count)
  for (node_i in seq_len(task_node_count)) {
    visit_rows[[node_i]] <- data.frame(
      trial_ids,
      node = node_i,
      visits = rowSums(node_mat == node_i, na.rm = TRUE),
      stringsAsFactors = FALSE
    )
  }
  visits <- do.call(rbind, visit_rows)
  node_rewards <- unique(dat[, unique(c(id_cols, "node", "actual_reward")), drop = FALSE])
  node_rewards$node <- suppressWarnings(as.integer(node_rewards$node))
  node_rewards$actual_reward <- suppressWarnings(as.numeric(node_rewards$actual_reward))
  visits <- merge(visits, node_rewards, by = c(id_cols, "node"), all.x = TRUE)
  visits <- visits[is.finite(visits$actual_reward), , drop = FALSE]
  if (nrow(visits) == 0) {
    return(data.frame())
  }
  summary_data <- aggregate(
    visits ~ beta + opportunity + node + actual_reward,
    data = visits,
    FUN = mean_or_na
  )
  count_data <- aggregate(
    visits ~ beta + opportunity + node + actual_reward,
    data = visits,
    FUN = length
  )
  names(count_data)[names(count_data) == "visits"] <- "n"
  merge(summary_data, count_data, by = c("beta", "opportunity", "node", "actual_reward"))
}

plot_node_visit_summary <- function(summary_data) {
  if (nrow(summary_data) == 0) {
    return(invisible(NULL))
  }
  path <- file.path(results_dir, sprintf("revisit_node_reward_vs_number_of_visits_%s.png", file_suffix))
  node_levels <- sort(unique(suppressWarnings(as.integer(summary_data$node))))
  n_cols <- min(3L, length(node_levels))
  n_rows <- ceiling(length(node_levels) / n_cols)
  open_panel_png(path, n_cols = n_cols, n_rows = n_rows, legend_fraction = legend_panel_fraction)
  old_par <- par(no.readonly = TRUE)
  panel_layout <- cbind(
    matrix(seq_len(n_rows * n_cols), nrow = n_rows, ncol = n_cols, byrow = TRUE),
    rep(n_rows * n_cols + 1L, n_rows)
  )
  layout(panel_layout, widths = c(rep(1, n_cols), legend_panel_fraction))
  par(mar = c(4.2, 4.2, 2, 1))
  apply_panel_text_style()
  x_limits <- expand_range(summary_data$actual_reward, pad = 0.08)
  y_limits <- expand_range(summary_data$visits, pad = 0.08)
  x_ticks <- seq(floor(x_limits[[1]]), ceiling(x_limits[[2]]), by = 1)
  for (panel_i in seq_len(n_rows * n_cols)) {
    if (panel_i > length(node_levels)) {
      plot.new()
      next
    }
    node_i <- node_levels[[panel_i]]
    plot(
      NA,
      xlim = x_limits,
      ylim = y_limits,
      xlab = "Node reward",
      ylab = "Average number of visits",
      main = sprintf("Node %d", node_i),
      xaxt = "n"
    )
    axis(1, at = x_ticks)
    grid()
    for (opportunity_value in opportunity_levels) {
      for (beta_value in beta_levels) {
        piece <- summary_data[
          summary_data$node == node_i &
            summary_data$beta == beta_value &
            summary_data$opportunity == opportunity_value,
          ,
          drop = FALSE
        ]
        if (nrow(piece) == 0) {
          next
        }
        piece <- piece[order(piece$actual_reward), , drop = FALSE]
        lines(
          piece$actual_reward,
          piece$visits,
          type = "b",
          pch = 19,
          lwd = 1.5,
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

build_timestep_best_summary <- function(trial_data) {
  if (length(observation_timesteps) == 0) {
    return(data.frame())
  }
  n_trials <- nrow(trial_data)
  row_index <- seq_len(n_trials)
  path_values <- matrix(0, nrow = n_trials, ncol = task_path_count)
  seen_nodes <- matrix(FALSE, nrow = n_trials, ncol = task_node_count)
  rows <- list()
  for (observation_timestep in observation_timesteps) {
    node_col <- paste0("expanded_node_t", observation_timestep)
    reward_col <- paste0("expanded_reward_t", observation_timestep)
    kl_col <- paste0("kl_d_t", observation_timestep)
    if (!all(c(node_col, reward_col) %in% names(trial_data))) {
      next
    }
    node_values <- suppressWarnings(as.integer(trial_data[[node_col]]))
    reward_values_t <- suppressWarnings(as.numeric(trial_data[[reward_col]]))
    path_indices <- path_id_for_node(node_values)
    valid_obs <- !is.na(node_values) &
      !is.na(path_indices) &
      !is.na(reward_values_t) &
      node_values >= 1L &
      node_values <= task_node_count &
      path_indices >= 1L &
      path_indices <= task_path_count
    if (any(valid_obs)) {
      node_idx <- cbind(row_index[valid_obs], node_values[valid_obs])
      new_node <- valid_obs
      new_node[valid_obs] <- !seen_nodes[node_idx]
      if (any(new_node)) {
        path_idx <- cbind(row_index[new_node], path_indices[new_node])
        path_values[path_idx] <- path_values[path_idx] + reward_values_t[new_node]
      }
      seen_nodes[node_idx] <- TRUE
    }
    keep <- valid_obs
    if (!any(keep)) {
      next
    }
    best_values <- apply(path_values, 1L, max)
    kl_values <- if (kl_col %in% names(trial_data)) {
      suppressWarnings(as.numeric(trial_data[[kl_col]]))
    } else {
      rep(NA_real_, n_trials)
    }
    rows[[length(rows) + 1L]] <- data.frame(
      beta = trial_data$beta[keep],
      opportunity = trial_data$opportunity[keep],
      observation_timestep = observation_timestep,
      best_path_value = best_values[keep],
      observations_before_stop = trial_data$observations_before_stop[keep],
      kl_paid = kl_values[keep],
      stringsAsFactors = FALSE
    )
  }
  if (length(rows) == 0) {
    return(data.frame())
  }
  dat <- do.call(rbind, rows)
  time_summary <- aggregate(
    observations_before_stop ~ beta + opportunity + observation_timestep + best_path_value,
    data = dat,
    FUN = mean_or_na
  )
  kl_summary <- aggregate(
    kl_paid ~ beta + opportunity + observation_timestep + best_path_value,
    data = dat,
    FUN = mean_or_na
  )
  count_data <- aggregate(
    observations_before_stop ~ beta + opportunity + observation_timestep + best_path_value,
    data = dat,
    FUN = length
  )
  names(count_data)[names(count_data) == "observations_before_stop"] <- "n"
  out <- merge(time_summary, kl_summary, by = c("beta", "opportunity", "observation_timestep", "best_path_value"), all = TRUE)
  merge(out, count_data, by = c("beta", "opportunity", "observation_timestep", "best_path_value"), all = TRUE)
}

plot_timestep_best_summary <- function(summary_data, y_col, ylab, file_prefix) {
  if (nrow(summary_data) == 0 || !y_col %in% names(summary_data)) {
    return(invisible(NULL))
  }
  summary_data <- summary_data[is.finite(summary_data[[y_col]]), , drop = FALSE]
  if (nrow(summary_data) == 0) {
    return(invisible(NULL))
  }
  path <- file.path(results_dir, sprintf("%s_%s.png", file_prefix, file_suffix))
  timestep_levels <- sort(unique(suppressWarnings(as.integer(summary_data$observation_timestep))))
  n_cols <- min(3L, length(timestep_levels))
  n_rows <- ceiling(length(timestep_levels) / n_cols)
  open_panel_png(path, n_cols = n_cols, n_rows = n_rows, legend_fraction = legend_panel_fraction)
  old_par <- par(no.readonly = TRUE)
  panel_layout <- cbind(
    matrix(seq_len(n_rows * n_cols), nrow = n_rows, ncol = n_cols, byrow = TRUE),
    rep(n_rows * n_cols + 1L, n_rows)
  )
  layout(panel_layout, widths = c(rep(1, n_cols), legend_panel_fraction))
  par(mar = c(4.2, 4.2, 2, 1))
  apply_panel_text_style()
  x_limits <- expand_range(summary_data$best_path_value, pad = 0.08)
  y_limits <- expand_range(summary_data[[y_col]], pad = 0.08)
  x_ticks <- seq(floor(x_limits[[1]]), ceiling(x_limits[[2]]), by = 1)
  for (panel_i in seq_len(n_rows * n_cols)) {
    if (panel_i > length(timestep_levels)) {
      plot.new()
      next
    }
    observation_timestep <- timestep_levels[[panel_i]]
    plot(
      NA,
      xlim = x_limits,
      ylim = y_limits,
      xlab = "Best path value observed so far",
      ylab = ylab,
      main = sprintf("Observation t%d", observation_timestep),
      xaxt = "n"
    )
    axis(1, at = x_ticks)
    grid()
    for (opportunity_value in opportunity_levels) {
      for (beta_value in beta_levels) {
        piece <- summary_data[
          summary_data$observation_timestep == observation_timestep &
            summary_data$beta == beta_value &
            summary_data$opportunity == opportunity_value,
          ,
          drop = FALSE
        ]
        if (nrow(piece) == 0) {
          next
        }
        piece <- piece[order(piece$best_path_value), , drop = FALSE]
        lines(
          piece$best_path_value,
          piece[[y_col]],
          type = "b",
          pch = 19,
          lwd = 1.5,
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

node_visit_summary <- build_node_visit_summary(all_data, trial_data, node_mat)
plot_node_visit_summary(node_visit_summary)

timestep_best_summary <- build_timestep_best_summary(trial_data)
plot_timestep_best_summary(
  timestep_best_summary,
  y_col = "observations_before_stop",
  ylab = "Average timestep before stopping",
  file_prefix = "revisit_timestep_before_stop_by_best_observed_path_value_by_observation_timestep"
)
plot_timestep_best_summary(
  timestep_best_summary,
  y_col = "kl_paid",
  ylab = "Average KL paid at timestep",
  file_prefix = "revisit_kl_paid_by_best_observed_path_value_by_observation_timestep"
)
