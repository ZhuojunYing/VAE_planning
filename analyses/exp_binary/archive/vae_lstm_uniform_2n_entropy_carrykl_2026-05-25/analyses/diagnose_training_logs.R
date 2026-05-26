#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)

get_arg <- function(i, default) {
  if (length(args) >= i && nzchar(args[[i]])) args[[i]] else default
}

beta_arg <- get_arg(1, "0.1,1.0,10.0,100.0,1000.0")
lambda_arg <- get_arg(2, "100.0")
alpha_arg <- get_arg(3, "0.0")
opportunity_arg <- get_arg(4, "0.0")
model_dir <- get_arg(5, "outputs/models")
results_dir <- get_arg(6, "results")
tree_size <- as.integer(get_arg(7, "2"))
expansion_decision_version <- get_arg(8, "lstm")
seed_arg <- get_arg(9, "1:1")
model_variant <- get_arg(10, "vae")

dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)

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

expansion_decision_version <- normalize_expansion_decision_version(expansion_decision_version)

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

model_variant <- normalize_model_variant(model_variant)

model_variant_file_segment <- function(variant) {
  sprintf("variant_%s_", variant)
}

model_variant_file_segments <- function(variant) {
  segments <- model_variant_file_segment(variant)
  if (identical(variant, "vae")) {
    # Backward compatibility for older VAE logs that predated explicit
    # variant_vae_ filename segments.
    segments <- c(segments, "")
  }
  unique(segments)
}

parse_list <- function(x) trimws(strsplit(x, ",")[[1]])

parse_seeds <- function(x) {
  x <- trimws(x)
  if (grepl(":", x, fixed = TRUE)) {
    parts <- as.integer(strsplit(x, ":", fixed = TRUE)[[1]])
    return(seq(parts[[1]], parts[[2]]))
  }
  as.integer(parse_list(x))
}

arg_label <- function(values) {
  label <- paste(values, collapse = "_")
  gsub("[^A-Za-z0-9._-]+", "_", label)
}

value_candidates <- function(x) {
  x_chr <- as.character(x)
  x_num <- suppressWarnings(as.numeric(x_chr))
  candidates <- x_chr
  if (!is.na(x_num)) {
    candidates <- c(
      candidates,
      sprintf("%.1f", x_num),
      sprintf("%.2f", x_num),
      format(x_num, scientific = FALSE, trim = TRUE)
    )
  }
  unique(candidates)
}

drop_unnamed_index_columns <- function(dat) {
  unnamed_cols <- names(dat) %in% c("", "...1", "X", "X1")
  if (any(unnamed_cols)) {
    dat <- dat[, !unnamed_cols, drop = FALSE]
  }
  dat
}

training_log_path <- function(lambda_value, alpha_value, beta_value, opportunity_value, seed) {
  for (lambda_candidate in value_candidates(lambda_value)) {
    for (alpha_candidate in value_candidates(alpha_value)) {
      for (beta_candidate in value_candidates(beta_value)) {
        for (opportunity_candidate in value_candidates(opportunity_value)) {
          for (variant_file_segment in model_variant_file_segments(model_variant)) {
            file_name <- sprintf(
              "lambda_%s_alpha_%s_beta_%s_opportunity_%s_expansion_%s_%sseed_%d_%dn_training_logs.csv",
              lambda_candidate,
              alpha_candidate,
              beta_candidate,
              opportunity_candidate,
              expansion_decision_version,
              variant_file_segment,
              seed,
              tree_size
            )
            file_path <- file.path(model_dir, file_name)
            if (file.exists(file_path)) {
              return(file_path)
            }
          }
        }
      }
    }
  }
  NA_character_
}

read_training_log <- function(beta_value, opportunity_value, seed) {
  file_path <- training_log_path(lambda_arg, alpha_arg, beta_value, opportunity_value, seed)
  if (is.na(file_path)) {
    warning(sprintf(
      "Missing training log for beta=%s opportunity=%s seed=%d model_variant=%s",
      beta_value,
      opportunity_value,
      seed,
      model_variant
    ))
    return(NULL)
  }

  dat <- read.csv(file_path, stringsAsFactors = FALSE)
  dat <- drop_unnamed_index_columns(dat)
  dat$beta <- beta_value
  dat$opportunity <- opportunity_value
  dat$seed <- seed
  dat$model_variant <- model_variant
  dat$file_path <- file_path
  dat$modified_time <- file.info(file_path)$mtime
  dat
}

beta_values <- parse_list(beta_arg)
opportunity_values <- parse_list(opportunity_arg)
seeds <- parse_seeds(seed_arg)

loaded_logs <- list()
for (beta_value in beta_values) {
  for (opportunity_value in opportunity_values) {
    for (seed in seeds) {
      dat <- read_training_log(beta_value, opportunity_value, seed)
      if (!is.null(dat)) {
        loaded_logs[[length(loaded_logs) + 1]] <- dat
      }
    }
  }
}

all_logs <- if (length(loaded_logs) > 0) do.call(rbind, loaded_logs) else NULL
if (is.null(all_logs) || nrow(all_logs) == 0) {
  stop("No training logs were found. Check beta/lambda/alpha/opportunity/expansion/seed values.")
}

to_numeric_column <- function(dat, col) {
  if (col %in% names(dat)) {
    suppressWarnings(as.numeric(dat[[col]]))
  } else {
    rep(NA_real_, nrow(dat))
  }
}

metric_cols <- intersect(
  c(
    "total_loss", "kl_loss", "action_loss", "reconstruction_loss",
    "expansion_loss", "critic_loss", "unified_decision_ce_loss",
    "expansion_stop_rate", "expansion_continue_rate",
    "action_p_correct_after_neg4_t1", "continue_rate_after_neg4_t1", "n_neg4_t1",
    "action_p_correct_after_pos4_t1", "continue_rate_after_pos4_t1", "n_pos4_t1",
    "terminal_prob_after_neg4_t1", "correct_terminal_prob_after_neg4_t1",
    "expand_unobserved_prob_after_neg4_t1", "expand_observed_prob_after_neg4_t1",
    "terminal_prob_after_pos4_t1", "correct_terminal_prob_after_pos4_t1",
    "expand_unobserved_prob_after_pos4_t1", "expand_observed_prob_after_pos4_t1",
    "sampled_return_stop_after_neg4_t1", "sampled_return_continue_after_neg4_t1",
    "sampled_return_stop_after_pos4_t1", "sampled_return_continue_after_pos4_t1",
    "kl_d_stop_after_neg4_t1", "kl_d_continue_after_neg4_t1",
    "kl_d_stop_after_pos4_t1", "kl_d_continue_after_pos4_t1",
    "n_stop_after_neg4_t1", "n_continue_after_neg4_t1",
    "n_stop_after_pos4_t1", "n_continue_after_pos4_t1",
    "kl_grad_norm_enc", "kl_grad_norm_lstm", "kl_grad_norm_dec",
    "act_grad_norm_enc", "act_grad_norm_lstm", "act_grad_norm_dec",
    "rec_grad_norm_enc", "rec_grad_norm_lstm", "rec_grad_norm_dec",
    "kl_grad_norm_prior", "act_grad_norm_prior", "rec_grad_norm_prior",
    "exp_grad_norm_enc", "exp_grad_norm_lstm", "exp_grad_norm_dec",
    "exp_grad_norm_prior", "exp_grad_norm_head", "exp_policy_grad_norm_head",
    "opp_grad_norm_enc", "opp_grad_norm_lstm", "opp_grad_norm_dec",
    "opp_grad_norm_prior", "opp_grad_norm_head",
    "critic_grad_norm_enc", "critic_grad_norm_lstm", "critic_grad_norm_dec",
    "critic_grad_norm_prior", "critic_grad_norm_head",
    "act_grad_norm_head", "rec_grad_norm_head",
    "update_grad_norm_enc", "update_grad_norm_lstm", "update_grad_norm_dec",
    "update_grad_norm_prior",
    "critic_pred_after_neg4_t1", "critic_pred_after_pos4_t1",
    "expansion_epsilon", "expansion_temperature", "expansion_entropy_coef",
    "learning_rate"
  ),
  names(all_logs)
)
metric_cols <- unique(c(metric_cols, grep("^lstm_probe_", names(all_logs), value = TRUE)))

for (col in metric_cols) {
  all_logs[[col]] <- suppressWarnings(as.numeric(all_logs[[col]]))
}
all_logs$epoch <- suppressWarnings(as.numeric(all_logs$epoch))

mean_or_na <- function(x) {
  x <- x[is.finite(x)]
  if (length(x) == 0) NA_real_ else mean(x)
}

safe_neg_log <- function(x) {
  x <- suppressWarnings(as.numeric(x))
  out <- -log(pmax(x, 1e-7))
  out[!is.finite(out)] <- NA_real_
  out
}

first_present_column <- function(dat, candidates) {
  present <- candidates[candidates %in% names(dat)]
  if (length(present) == 0) NULL else present[[1]]
}

copy_optional_metric <- function(dat, output_col, candidates) {
  source_col <- first_present_column(dat, candidates)
  if (is.null(source_col)) {
    dat[[output_col]] <- NA_real_
  } else {
    dat[[output_col]] <- suppressWarnings(as.numeric(dat[[source_col]]))
  }
  dat
}

aggregate_metric <- function(dat, value_col) {
  if (!value_col %in% names(dat)) {
    return(data.frame())
  }
  aggregate(
    dat[[value_col]],
    by = list(
      beta = dat$beta,
      opportunity = dat$opportunity,
      epoch = dat$epoch
    ),
    FUN = mean_or_na
  ) |>
    stats::setNames(c("beta", "opportunity", "epoch", "value"))
}

palette_for <- function(levels) {
  cols <- grDevices::hcl.colors(max(3, length(levels) + 2), palette = "Blues")
  setNames(cols[seq_along(levels)], levels)
}

beta_levels <- beta_values[beta_values %in% unique(all_logs$beta)]
if (length(beta_levels) == 0) beta_levels <- sort(unique(all_logs$beta))
opportunity_levels <- opportunity_values[opportunity_values %in% unique(all_logs$opportunity)]
if (length(opportunity_levels) == 0) opportunity_levels <- sort(unique(all_logs$opportunity))

color_by <- if (length(opportunity_levels) > 1 && length(beta_levels) == 1) {
  "opportunity"
} else {
  "beta"
}
if (length(beta_levels) > 1 && length(opportunity_levels) > 1) {
  warning("Both beta and opportunity have multiple values; using beta for color and opportunity for line style.")
}

color_levels <- if (identical(color_by, "opportunity")) opportunity_levels else beta_levels
series_cols <- palette_for(color_levels)
opportunity_lty_values <- c(1, 2, 3, 4, 5, 6)
opportunity_lty <- setNames(
  rep(opportunity_lty_values, length.out = length(opportunity_levels)),
  opportunity_levels
)

series_color <- function(beta_value, opportunity_value) {
  if (identical(color_by, "opportunity")) {
    series_cols[[as.character(opportunity_value)]]
  } else {
    series_cols[[as.character(beta_value)]]
  }
}

plot_metric <- function(dat, value_col, ylab, log_y = FALSE) {
  summary_dat <- aggregate_metric(dat, value_col)
  if (nrow(summary_dat) == 0 || all(!is.finite(summary_dat$value))) {
    plot(NA, xlim = c(0, 1), ylim = c(0, 1), xlab = "Epoch", ylab = ylab, main = "")
    text(0.5, 0.5, sprintf("No %s column", value_col))
    return(invisible(NULL))
  }

  y_values <- summary_dat$value
  y_values <- y_values[is.finite(y_values)]
  if (log_y) {
    y_values <- y_values[y_values > 0]
  }
  if (length(y_values) == 0) {
    plot(NA, xlim = c(0, 1), ylim = c(0, 1), xlab = "Epoch", ylab = ylab, main = "")
    text(0.5, 0.5, sprintf("No finite positive %s values", value_col))
    return(invisible(NULL))
  }
  x_values <- summary_dat$epoch[is.finite(summary_dat$epoch)]
  x_range <- range(x_values, finite = TRUE)
  y_range <- range(y_values, finite = TRUE)
  if (!all(is.finite(x_range)) || !all(is.finite(y_range))) {
    plot(NA, xlim = c(0, 1), ylim = c(0, 1), xlab = "Epoch", ylab = ylab, main = "")
    text(0.5, 0.5, sprintf("No finite %s values", value_col))
    return(invisible(NULL))
  }
  if (identical(x_range[[1]], x_range[[2]])) {
    x_range <- x_range + c(-0.5, 0.5)
  }
  if (identical(y_range[[1]], y_range[[2]])) {
    y_pad <- if (y_range[[1]] == 0) 0.5 else abs(y_range[[1]]) * 0.05
    y_range <- y_range + c(-y_pad, y_pad)
    if (log_y && y_range[[1]] <= 0) {
      y_range[[1]] <- max(y_range[[2]] * 0.1, .Machine$double.eps)
    }
  }

  plot(
    NA,
    xlim = x_range,
    ylim = y_range,
    log = if (log_y) "y" else "",
    xlab = "Epoch",
    ylab = ylab,
    main = ""
  )
  grid()

  for (opportunity_value in opportunity_levels) {
    for (beta_value in beta_levels) {
      series <- summary_dat[
        summary_dat$beta == beta_value & summary_dat$opportunity == opportunity_value,
        ,
        drop = FALSE
      ]
      series <- series[order(series$epoch), , drop = FALSE]
      if (nrow(series) > 0) {
        lines(
          series$epoch,
          series$value,
          col = series_color(beta_value, opportunity_value),
          lwd = 2,
          lty = opportunity_lty[[as.character(opportunity_value)]]
        )
      }
    }
  }
}

add_beta_legend <- function() {
  old_xpd <- par("xpd")
  par(xpd = NA)
  legend(
    "topright",
    inset = c(-0.35, 0),
    legend = paste(color_by, color_levels),
    col = series_cols[color_levels],
    lwd = 2,
    bty = "n"
  )
  if (length(opportunity_levels) > 1 && !identical(color_by, "opportunity")) {
    legend(
      "bottomright",
      inset = c(-0.35, 0),
      legend = paste("opportunity", opportunity_levels),
      lty = opportunity_lty[opportunity_levels],
      col = "black",
      lwd = 2,
      bty = "n"
    )
  }
  par(xpd = old_xpd)
}

safe_ratio <- function(num, den) {
  ratio <- num / den
  ratio[!is.finite(ratio)] <- NA_real_
  ratio
}

all_logs$act_to_kl_grad_enc <- safe_ratio(
  to_numeric_column(all_logs, "act_grad_norm_enc"),
  to_numeric_column(all_logs, "kl_grad_norm_enc")
)
all_logs$act_to_kl_grad_lstm <- safe_ratio(
  to_numeric_column(all_logs, "act_grad_norm_lstm"),
  to_numeric_column(all_logs, "kl_grad_norm_lstm")
)
all_logs$act_to_kl_grad_dec <- safe_ratio(
  to_numeric_column(all_logs, "act_grad_norm_dec"),
  to_numeric_column(all_logs, "kl_grad_norm_dec")
)

# Approximate boundary-specific decision CE from currently logged mean
# probabilities. This is -log(mean p_correct_terminal), not mean(-log p)
# over individual trials, so use it as a directionally useful diagnostic only.
all_logs$approx_decision_ce_after_neg4_t1 <- safe_neg_log(
  to_numeric_column(all_logs, "correct_terminal_prob_after_neg4_t1")
)
all_logs$approx_decision_ce_after_pos4_t1 <- safe_neg_log(
  to_numeric_column(all_logs, "correct_terminal_prob_after_pos4_t1")
)

# These columns are populated when newer training logs record exact
# boundary-specific quantities. Older logs will show blank panels with an
# explanatory message.
all_logs <- copy_optional_metric(
  all_logs,
  "decision_ce_after_neg4_t1",
  c("decision_ce_after_neg4_t1", "unified_decision_ce_after_neg4_t1", "ce_after_neg4_t1")
)
all_logs <- copy_optional_metric(
  all_logs,
  "decision_ce_after_pos4_t1",
  c("decision_ce_after_pos4_t1", "unified_decision_ce_after_pos4_t1", "ce_after_pos4_t1")
)
all_logs <- copy_optional_metric(
  all_logs,
  "chosen_path_reward_after_neg4_t1",
  c("chosen_path_reward_after_neg4_t1", "selected_path_reward_after_neg4_t1", "episode_reward_after_neg4_t1")
)
all_logs <- copy_optional_metric(
  all_logs,
  "chosen_path_reward_after_pos4_t1",
  c("chosen_path_reward_after_pos4_t1", "selected_path_reward_after_pos4_t1", "episode_reward_after_pos4_t1")
)
all_logs <- copy_optional_metric(
  all_logs,
  "kl_d_after_neg4_t1",
  c("kl_d_after_neg4_t1", "kl_after_neg4_t1", "first_kl_after_neg4_t1")
)
all_logs <- copy_optional_metric(
  all_logs,
  "kl_d_after_pos4_t1",
  c("kl_d_after_pos4_t1", "kl_after_pos4_t1", "first_kl_after_pos4_t1")
)
all_logs <- copy_optional_metric(
  all_logs,
  "critic_pred_after_neg4_t1",
  c("critic_pred_after_neg4_t1", "critic_value_after_neg4_t1", "value_pred_after_neg4_t1")
)
all_logs <- copy_optional_metric(
  all_logs,
  "critic_pred_after_pos4_t1",
  c("critic_pred_after_pos4_t1", "critic_value_after_pos4_t1", "value_pred_after_pos4_t1")
)
all_logs <- copy_optional_metric(
  all_logs,
  "sampled_return_stop_after_neg4_t1",
  c("sampled_return_stop_after_neg4_t1", "return_stop_after_neg4_t1")
)
all_logs <- copy_optional_metric(
  all_logs,
  "sampled_return_continue_after_neg4_t1",
  c("sampled_return_continue_after_neg4_t1", "return_continue_after_neg4_t1")
)
all_logs <- copy_optional_metric(
  all_logs,
  "sampled_return_stop_after_pos4_t1",
  c("sampled_return_stop_after_pos4_t1", "return_stop_after_pos4_t1")
)
all_logs <- copy_optional_metric(
  all_logs,
  "sampled_return_continue_after_pos4_t1",
  c("sampled_return_continue_after_pos4_t1", "return_continue_after_pos4_t1")
)
all_logs <- copy_optional_metric(
  all_logs,
  "kl_d_stop_after_neg4_t1",
  c("kl_d_stop_after_neg4_t1", "kl_stop_after_neg4_t1")
)
all_logs <- copy_optional_metric(
  all_logs,
  "kl_d_continue_after_neg4_t1",
  c("kl_d_continue_after_neg4_t1", "kl_continue_after_neg4_t1")
)
all_logs <- copy_optional_metric(
  all_logs,
  "kl_d_stop_after_pos4_t1",
  c("kl_d_stop_after_pos4_t1", "kl_stop_after_pos4_t1")
)
all_logs <- copy_optional_metric(
  all_logs,
  "kl_d_continue_after_pos4_t1",
  c("kl_d_continue_after_pos4_t1", "kl_continue_after_pos4_t1")
)
all_logs$return_stop_minus_continue_after_neg4_t1 <- (
  to_numeric_column(all_logs, "sampled_return_stop_after_neg4_t1") -
    to_numeric_column(all_logs, "sampled_return_continue_after_neg4_t1")
)
all_logs$return_stop_minus_continue_after_pos4_t1 <- (
  to_numeric_column(all_logs, "sampled_return_stop_after_pos4_t1") -
    to_numeric_column(all_logs, "sampled_return_continue_after_pos4_t1")
)
all_logs$kl_stop_minus_continue_after_neg4_t1 <- (
  to_numeric_column(all_logs, "kl_d_stop_after_neg4_t1") -
    to_numeric_column(all_logs, "kl_d_continue_after_neg4_t1")
)
all_logs$kl_stop_minus_continue_after_pos4_t1 <- (
  to_numeric_column(all_logs, "kl_d_stop_after_pos4_t1") -
    to_numeric_column(all_logs, "kl_d_continue_after_pos4_t1")
)

probe_reward_values <- c(-4, -3, -2, -1, 0, 1, 2, 3, 4)
probe_reward_label <- function(value) {
  if (value > 0) return(sprintf("p%d", value))
  if (value < 0) return(sprintf("m%d", abs(value)))
  "z0"
}

diagnostic_label <- sprintf(
  "lambda_%s_alpha_%s_beta_%s_opportunity_%s_expansion_%s_%s%dn",
  lambda_arg,
  alpha_arg,
  arg_label(beta_values),
  arg_label(opportunity_values),
  expansion_decision_version,
  model_variant_file_segment(model_variant),
  tree_size
)

pdf_path <- file.path(results_dir, sprintf("training_diagnostics_%s.pdf", diagnostic_label))
pdf(pdf_path, width = 21, height = 30)
old_par <- par(mfrow = c(10, 4), mar = c(4.5, 4.5, 1, 7))

if ("expansion_epsilon" %in% names(all_logs)) {
  plot_metric(all_logs, "expansion_epsilon", "Expansion epsilon")
} else if ("expansion_temperature" %in% names(all_logs)) {
  plot_metric(all_logs, "expansion_temperature", "Expansion temperature")
} else if ("expansion_entropy_coef" %in% names(all_logs)) {
  plot_metric(all_logs, "expansion_entropy_coef", "Expansion entropy coef")
} else {
  plot(NA, xlab = "Epoch", ylab = "Exploration schedule", main = "")
  text(0.5, 0.5, "No exploration schedule column")
}
plot_metric(all_logs, "expansion_entropy_coef", "Expansion entropy coef")

plot_metric(all_logs, "expansion_stop_rate", "Expansion stop rate")
plot_metric(all_logs, "expansion_continue_rate", "Expansion continue rate")
plot_metric(all_logs, "action_p_correct_after_neg4_t1", "P(action correct | first reward -4)")
plot_metric(all_logs, "continue_rate_after_neg4_t1", "P(continue | first reward -4)")
plot_metric(all_logs, "action_p_correct_after_pos4_t1", "P(action correct | first reward 4)")
plot_metric(all_logs, "continue_rate_after_pos4_t1", "P(continue | first reward 4)")
plot_metric(all_logs, "terminal_prob_after_neg4_t1", "P(terminal mass | first reward -4)")
plot_metric(all_logs, "correct_terminal_prob_after_neg4_t1", "P(correct terminal | first reward -4)")
plot_metric(all_logs, "expand_unobserved_prob_after_neg4_t1", "P(expand unobserved | first reward -4)")
plot_metric(all_logs, "expand_observed_prob_after_neg4_t1", "P(expand observed | first reward -4)")
plot_metric(all_logs, "terminal_prob_after_pos4_t1", "P(terminal mass | first reward 4)")
plot_metric(all_logs, "correct_terminal_prob_after_pos4_t1", "P(correct terminal | first reward 4)")
plot_metric(all_logs, "expand_unobserved_prob_after_pos4_t1", "P(expand unobserved | first reward 4)")
plot_metric(all_logs, "expand_observed_prob_after_pos4_t1", "P(expand observed | first reward 4)")
plot_metric(all_logs, "approx_decision_ce_after_neg4_t1", "Approx CE | first reward -4")
plot_metric(all_logs, "approx_decision_ce_after_pos4_t1", "Approx CE | first reward 4")
plot_metric(all_logs, "decision_ce_after_neg4_t1", "Exact decision CE | first reward -4")
plot_metric(all_logs, "decision_ce_after_pos4_t1", "Exact decision CE | first reward 4")
plot_metric(all_logs, "chosen_path_reward_after_neg4_t1", "Chosen path reward | first reward -4")
plot_metric(all_logs, "chosen_path_reward_after_pos4_t1", "Chosen path reward | first reward 4")
plot_metric(all_logs, "kl_d_after_neg4_t1", "KL_d | first reward -4")
plot_metric(all_logs, "kl_d_after_pos4_t1", "KL_d | first reward 4")
plot_metric(all_logs, "critic_pred_after_neg4_t1", "Critic pred | first reward -4")
plot_metric(all_logs, "critic_pred_after_pos4_t1", "Critic pred | first reward 4")
plot_metric(all_logs, "sampled_return_stop_after_neg4_t1", "Return | stop after -4")
plot_metric(all_logs, "sampled_return_continue_after_neg4_t1", "Return | continue after -4")
plot_metric(all_logs, "return_stop_minus_continue_after_neg4_t1", "Return stop - continue | -4")
plot_metric(all_logs, "kl_d_stop_after_neg4_t1", "KL_d | stop after -4")
plot_metric(all_logs, "kl_d_continue_after_neg4_t1", "KL_d | continue after -4")
plot_metric(all_logs, "kl_stop_minus_continue_after_neg4_t1", "KL stop - continue | -4")
plot_metric(all_logs, "sampled_return_stop_after_pos4_t1", "Return | stop after 4")
plot_metric(all_logs, "sampled_return_continue_after_pos4_t1", "Return | continue after 4")
plot_metric(all_logs, "return_stop_minus_continue_after_pos4_t1", "Return stop - continue | 4")
plot_metric(all_logs, "kl_d_stop_after_pos4_t1", "KL_d | stop after 4")
plot_metric(all_logs, "kl_d_continue_after_pos4_t1", "KL_d | continue after 4")
plot_metric(all_logs, "kl_stop_minus_continue_after_pos4_t1", "KL stop - continue | 4")
plot_metric(all_logs, "total_loss", "Total loss")
plot_metric(all_logs, "expansion_loss", "Expansion loss")
plot_metric(all_logs, "critic_loss", "Critic loss")
plot_metric(all_logs, "unified_decision_ce_loss", "Unified decision CE loss")
plot_metric(all_logs, "action_loss", "Action loss")
plot_metric(all_logs, "kl_loss", "KL loss")
plot_metric(all_logs, "reconstruction_loss", "Reconstruction loss")
plot_metric(all_logs, "lstm_probe_accuracy", "LSTM reward probe accuracy")
plot_metric(all_logs, "lstm_probe_loss", "LSTM reward probe loss")
plot_metric(all_logs, "learning_rate", "Learning rate", log_y = TRUE)
plot_metric(all_logs, "act_to_kl_grad_enc", "Action/KL grad ratio encoder", log_y = TRUE)
plot_metric(all_logs, "act_to_kl_grad_lstm", "Action/KL grad ratio LSTM", log_y = TRUE)
plot_metric(all_logs, "act_to_kl_grad_dec", "Action/KL grad ratio decoder", log_y = TRUE)

add_beta_legend()
par(old_par)
dev.off()

grad_pdf_path <- file.path(results_dir, sprintf("training_gradients_%s.pdf", diagnostic_label))
pdf(grad_pdf_path, width = 18, height = 25)
old_par <- par(mfrow = c(7, 4), mar = c(4.5, 4.5, 1, 7))
plot_metric(all_logs, "update_grad_norm_enc", "Update grad encoder", log_y = TRUE)
plot_metric(all_logs, "update_grad_norm_lstm", "Update grad LSTM", log_y = TRUE)
plot_metric(all_logs, "update_grad_norm_dec", "Update grad decoder", log_y = TRUE)
plot_metric(all_logs, "update_grad_norm_prior", "Update grad prior", log_y = TRUE)
plot_metric(all_logs, "exp_policy_grad_norm_head", "Expansion policy grad head", log_y = TRUE)
plot_metric(all_logs, "exp_grad_norm_lstm", "Expansion policy grad LSTM", log_y = TRUE)
plot_metric(all_logs, "exp_grad_norm_enc", "Expansion policy grad encoder", log_y = TRUE)
plot_metric(all_logs, "exp_grad_norm_dec", "Expansion policy grad decoder", log_y = TRUE)
plot_metric(all_logs, "opp_grad_norm_head", "Opportunity grad head", log_y = TRUE)
plot_metric(all_logs, "opp_grad_norm_lstm", "Opportunity grad LSTM", log_y = TRUE)
plot_metric(all_logs, "opp_grad_norm_enc", "Opportunity grad encoder", log_y = TRUE)
plot_metric(all_logs, "opp_grad_norm_dec", "Opportunity grad decoder", log_y = TRUE)
plot_metric(all_logs, "act_grad_norm_head", "Action grad head", log_y = TRUE)
plot_metric(all_logs, "act_grad_norm_enc", "Action grad encoder", log_y = TRUE)
plot_metric(all_logs, "act_grad_norm_lstm", "Action grad LSTM", log_y = TRUE)
plot_metric(all_logs, "act_grad_norm_dec", "Action grad decoder", log_y = TRUE)
plot_metric(all_logs, "critic_grad_norm_head", "Critic grad head", log_y = TRUE)
plot_metric(all_logs, "critic_grad_norm_enc", "Critic grad encoder", log_y = TRUE)
plot_metric(all_logs, "critic_grad_norm_lstm", "Critic grad LSTM", log_y = TRUE)
plot_metric(all_logs, "critic_grad_norm_dec", "Critic grad decoder", log_y = TRUE)
plot_metric(all_logs, "kl_grad_norm_enc", "KL grad encoder", log_y = TRUE)
plot_metric(all_logs, "kl_grad_norm_lstm", "KL grad LSTM", log_y = TRUE)
plot_metric(all_logs, "kl_grad_norm_dec", "KL grad decoder", log_y = TRUE)
plot_metric(all_logs, "rec_grad_norm_head", "Recon grad head", log_y = TRUE)
plot_metric(all_logs, "lstm_probe_grad_norm_head", "LSTM probe grad head", log_y = TRUE)
plot_metric(all_logs, "rec_grad_norm_enc", "Recon grad encoder", log_y = TRUE)
plot_metric(all_logs, "rec_grad_norm_lstm", "Recon grad LSTM", log_y = TRUE)
plot_metric(all_logs, "rec_grad_norm_dec", "Recon grad decoder", log_y = TRUE)
add_beta_legend()
par(old_par)
dev.off()

probe_pdf_path <- file.path(results_dir, sprintf("training_lstm_reward_probe_%s.pdf", diagnostic_label))
pdf(probe_pdf_path, width = 18, height = 14)
old_par <- par(mfrow = c(3, 3), mar = c(4.5, 4.5, 1, 7))
for (reward_value in probe_reward_values) {
  label <- probe_reward_label(reward_value)
  plot_metric(
    all_logs,
    sprintf("lstm_probe_acc_reward_%s", label),
    sprintf("Probe accuracy | reward %s", reward_value)
  )
}
add_beta_legend()
for (reward_value in probe_reward_values) {
  label <- probe_reward_label(reward_value)
  plot_metric(
    all_logs,
    sprintf("lstm_probe_loss_reward_%s", label),
    sprintf("Probe loss | reward %s", reward_value),
    log_y = TRUE
  )
}
add_beta_legend()
for (reward_value in probe_reward_values) {
  label <- probe_reward_label(reward_value)
  plot_metric(
    all_logs,
    sprintf("lstm_probe_n_reward_%s", label),
    sprintf("Probe n | reward %s", reward_value)
  )
}
add_beta_legend()
par(old_par)
dev.off()

summary_rows <- do.call(rbind, lapply(split(all_logs, list(all_logs$beta, all_logs$opportunity, all_logs$seed), drop = TRUE), function(dat) {
  dat <- dat[order(dat$epoch), , drop = FALSE]
  last_row <- dat[nrow(dat), , drop = FALSE]
  first_row <- dat[1, , drop = FALSE]
  data.frame(
    beta = last_row$beta,
    opportunity = last_row$opportunity,
    seed = last_row$seed,
    n_epochs = nrow(dat),
    first_epsilon = if ("expansion_epsilon" %in% names(dat)) first_row$expansion_epsilon else NA_real_,
    final_epsilon = if ("expansion_epsilon" %in% names(dat)) last_row$expansion_epsilon else NA_real_,
    final_total_loss = if ("total_loss" %in% names(dat)) last_row$total_loss else NA_real_,
    final_action_loss = if ("action_loss" %in% names(dat)) last_row$action_loss else NA_real_,
    final_kl_loss = if ("kl_loss" %in% names(dat)) last_row$kl_loss else NA_real_,
    final_reconstruction_loss = if ("reconstruction_loss" %in% names(dat)) last_row$reconstruction_loss else NA_real_,
    final_expansion_loss = if ("expansion_loss" %in% names(dat)) last_row$expansion_loss else NA_real_,
    final_critic_loss = if ("critic_loss" %in% names(dat)) last_row$critic_loss else NA_real_,
    final_lstm_probe_loss = if ("lstm_probe_loss" %in% names(dat)) last_row$lstm_probe_loss else NA_real_,
    final_lstm_probe_accuracy = if ("lstm_probe_accuracy" %in% names(dat)) last_row$lstm_probe_accuracy else NA_real_,
    final_lstm_probe_acc_reward_m4 = if ("lstm_probe_acc_reward_m4" %in% names(dat)) last_row$lstm_probe_acc_reward_m4 else NA_real_,
    final_lstm_probe_acc_reward_p4 = if ("lstm_probe_acc_reward_p4" %in% names(dat)) last_row$lstm_probe_acc_reward_p4 else NA_real_,
    final_unified_decision_ce_loss = if ("unified_decision_ce_loss" %in% names(dat)) last_row$unified_decision_ce_loss else NA_real_,
    final_expansion_stop_rate = if ("expansion_stop_rate" %in% names(dat)) last_row$expansion_stop_rate else NA_real_,
    final_expansion_continue_rate = if ("expansion_continue_rate" %in% names(dat)) last_row$expansion_continue_rate else NA_real_,
    final_action_p_correct_after_neg4_t1 = if ("action_p_correct_after_neg4_t1" %in% names(dat)) last_row$action_p_correct_after_neg4_t1 else NA_real_,
    final_continue_rate_after_neg4_t1 = if ("continue_rate_after_neg4_t1" %in% names(dat)) last_row$continue_rate_after_neg4_t1 else NA_real_,
    final_terminal_prob_after_neg4_t1 = if ("terminal_prob_after_neg4_t1" %in% names(dat)) last_row$terminal_prob_after_neg4_t1 else NA_real_,
    final_correct_terminal_prob_after_neg4_t1 = if ("correct_terminal_prob_after_neg4_t1" %in% names(dat)) last_row$correct_terminal_prob_after_neg4_t1 else NA_real_,
    final_expand_unobserved_prob_after_neg4_t1 = if ("expand_unobserved_prob_after_neg4_t1" %in% names(dat)) last_row$expand_unobserved_prob_after_neg4_t1 else NA_real_,
    final_expand_observed_prob_after_neg4_t1 = if ("expand_observed_prob_after_neg4_t1" %in% names(dat)) last_row$expand_observed_prob_after_neg4_t1 else NA_real_,
    final_approx_decision_ce_after_neg4_t1 = if ("approx_decision_ce_after_neg4_t1" %in% names(dat)) last_row$approx_decision_ce_after_neg4_t1 else NA_real_,
    final_decision_ce_after_neg4_t1 = if ("decision_ce_after_neg4_t1" %in% names(dat)) last_row$decision_ce_after_neg4_t1 else NA_real_,
    final_chosen_path_reward_after_neg4_t1 = if ("chosen_path_reward_after_neg4_t1" %in% names(dat)) last_row$chosen_path_reward_after_neg4_t1 else NA_real_,
    final_kl_d_after_neg4_t1 = if ("kl_d_after_neg4_t1" %in% names(dat)) last_row$kl_d_after_neg4_t1 else NA_real_,
    final_critic_pred_after_neg4_t1 = if ("critic_pred_after_neg4_t1" %in% names(dat)) last_row$critic_pred_after_neg4_t1 else NA_real_,
    final_sampled_return_stop_after_neg4_t1 = if ("sampled_return_stop_after_neg4_t1" %in% names(dat)) last_row$sampled_return_stop_after_neg4_t1 else NA_real_,
    final_sampled_return_continue_after_neg4_t1 = if ("sampled_return_continue_after_neg4_t1" %in% names(dat)) last_row$sampled_return_continue_after_neg4_t1 else NA_real_,
    final_return_stop_minus_continue_after_neg4_t1 = if ("return_stop_minus_continue_after_neg4_t1" %in% names(dat)) last_row$return_stop_minus_continue_after_neg4_t1 else NA_real_,
    final_kl_d_stop_after_neg4_t1 = if ("kl_d_stop_after_neg4_t1" %in% names(dat)) last_row$kl_d_stop_after_neg4_t1 else NA_real_,
    final_kl_d_continue_after_neg4_t1 = if ("kl_d_continue_after_neg4_t1" %in% names(dat)) last_row$kl_d_continue_after_neg4_t1 else NA_real_,
    final_n_stop_after_neg4_t1 = if ("n_stop_after_neg4_t1" %in% names(dat)) last_row$n_stop_after_neg4_t1 else NA_real_,
    final_n_continue_after_neg4_t1 = if ("n_continue_after_neg4_t1" %in% names(dat)) last_row$n_continue_after_neg4_t1 else NA_real_,
    final_n_neg4_t1 = if ("n_neg4_t1" %in% names(dat)) last_row$n_neg4_t1 else NA_real_,
    final_action_p_correct_after_pos4_t1 = if ("action_p_correct_after_pos4_t1" %in% names(dat)) last_row$action_p_correct_after_pos4_t1 else NA_real_,
    final_continue_rate_after_pos4_t1 = if ("continue_rate_after_pos4_t1" %in% names(dat)) last_row$continue_rate_after_pos4_t1 else NA_real_,
    final_terminal_prob_after_pos4_t1 = if ("terminal_prob_after_pos4_t1" %in% names(dat)) last_row$terminal_prob_after_pos4_t1 else NA_real_,
    final_correct_terminal_prob_after_pos4_t1 = if ("correct_terminal_prob_after_pos4_t1" %in% names(dat)) last_row$correct_terminal_prob_after_pos4_t1 else NA_real_,
    final_expand_unobserved_prob_after_pos4_t1 = if ("expand_unobserved_prob_after_pos4_t1" %in% names(dat)) last_row$expand_unobserved_prob_after_pos4_t1 else NA_real_,
    final_expand_observed_prob_after_pos4_t1 = if ("expand_observed_prob_after_pos4_t1" %in% names(dat)) last_row$expand_observed_prob_after_pos4_t1 else NA_real_,
    final_approx_decision_ce_after_pos4_t1 = if ("approx_decision_ce_after_pos4_t1" %in% names(dat)) last_row$approx_decision_ce_after_pos4_t1 else NA_real_,
    final_decision_ce_after_pos4_t1 = if ("decision_ce_after_pos4_t1" %in% names(dat)) last_row$decision_ce_after_pos4_t1 else NA_real_,
    final_chosen_path_reward_after_pos4_t1 = if ("chosen_path_reward_after_pos4_t1" %in% names(dat)) last_row$chosen_path_reward_after_pos4_t1 else NA_real_,
    final_kl_d_after_pos4_t1 = if ("kl_d_after_pos4_t1" %in% names(dat)) last_row$kl_d_after_pos4_t1 else NA_real_,
    final_critic_pred_after_pos4_t1 = if ("critic_pred_after_pos4_t1" %in% names(dat)) last_row$critic_pred_after_pos4_t1 else NA_real_,
    final_sampled_return_stop_after_pos4_t1 = if ("sampled_return_stop_after_pos4_t1" %in% names(dat)) last_row$sampled_return_stop_after_pos4_t1 else NA_real_,
    final_sampled_return_continue_after_pos4_t1 = if ("sampled_return_continue_after_pos4_t1" %in% names(dat)) last_row$sampled_return_continue_after_pos4_t1 else NA_real_,
    final_return_stop_minus_continue_after_pos4_t1 = if ("return_stop_minus_continue_after_pos4_t1" %in% names(dat)) last_row$return_stop_minus_continue_after_pos4_t1 else NA_real_,
    final_kl_d_stop_after_pos4_t1 = if ("kl_d_stop_after_pos4_t1" %in% names(dat)) last_row$kl_d_stop_after_pos4_t1 else NA_real_,
    final_kl_d_continue_after_pos4_t1 = if ("kl_d_continue_after_pos4_t1" %in% names(dat)) last_row$kl_d_continue_after_pos4_t1 else NA_real_,
    final_n_stop_after_pos4_t1 = if ("n_stop_after_pos4_t1" %in% names(dat)) last_row$n_stop_after_pos4_t1 else NA_real_,
    final_n_continue_after_pos4_t1 = if ("n_continue_after_pos4_t1" %in% names(dat)) last_row$n_continue_after_pos4_t1 else NA_real_,
    final_n_pos4_t1 = if ("n_pos4_t1" %in% names(dat)) last_row$n_pos4_t1 else NA_real_,
    final_act_to_kl_grad_lstm = last_row$act_to_kl_grad_lstm,
    final_act_to_kl_grad_dec = last_row$act_to_kl_grad_dec,
    file_path = last_row$file_path,
    modified_time = last_row$modified_time,
    stringsAsFactors = FALSE
  )
}))

summary_path <- file.path(results_dir, sprintf("training_diagnostics_%s.csv", diagnostic_label))
write.csv(summary_rows, summary_path, row.names = FALSE)

cat("\nTraining-log diagnostic notes:\n")
if (!"expansion_epsilon" %in% names(all_logs)) {
  cat("- These logs do not contain expansion_epsilon, so they were likely produced before epsilon-greedy exploration was added.\n")
} else {
  cat(sprintf(
    "- expansion_epsilon range in loaded logs: %.4f to %.4f\n",
    min(all_logs$expansion_epsilon, na.rm = TRUE),
    max(all_logs$expansion_epsilon, na.rm = TRUE)
  ))
}
if (!all(c("expansion_loss", "expansion_stop_rate", "expansion_continue_rate") %in% names(all_logs))) {
  cat("- These logs do not contain all new expansion diagnostics; rerun training with the updated train.py to populate expansion_loss and stop/continue rates.\n")
}
if (!all(c("lstm_probe_accuracy", "lstm_probe_acc_reward_m4", "lstm_probe_acc_reward_p4") %in% names(all_logs))) {
  cat("- These logs do not contain LSTM reward probe diagnostics. Rerun training with the updated train.py to populate the probe panels.\n")
}
if (!all(c("terminal_prob_after_neg4_t1", "terminal_prob_after_pos4_t1") %in% names(all_logs))) {
  cat("- These logs do not contain unified-policy probability mass diagnostics. Add terminal_prob_after_* and expand_*_prob_after_* columns during training to populate those panels.\n")
}
cat("- Approx CE panels use -log(mean correct-terminal probability), so they are not exact mean per-trial CE.\n")
if (all(!is.finite(all_logs$decision_ce_after_neg4_t1)) && all(!is.finite(all_logs$decision_ce_after_pos4_t1))) {
  cat("- Exact boundary-specific decision CE is not in these logs. Add decision_ce_after_* columns during training to populate those panels.\n")
}
if (all(!is.finite(all_logs$chosen_path_reward_after_neg4_t1)) && all(!is.finite(all_logs$chosen_path_reward_after_pos4_t1))) {
  cat("- Boundary-specific chosen path rewards are not in these logs. Add chosen_path_reward_after_* columns during training to populate those panels.\n")
}
if (all(!is.finite(all_logs$kl_d_after_neg4_t1)) && all(!is.finite(all_logs$kl_d_after_pos4_t1))) {
  cat("- Boundary-specific KL_d is not in these logs. Add kl_d_after_* columns during training to populate those panels.\n")
}
if (all(!is.finite(all_logs$critic_pred_after_neg4_t1)) && all(!is.finite(all_logs$critic_pred_after_pos4_t1))) {
  cat("- Boundary-specific critic predictions are not in these logs. Add critic_pred_after_* columns during training to populate those panels.\n")
}
cat("- Use the action/KL gradient ratio panels to see whether the action signal overwhelms the information signal in the shared encoder/LSTM/decoder.\n")
cat("- Use the first panel to confirm that epsilon was present for enough epochs and did not start/end at zero.\n")
cat("\nWrote: ", pdf_path, "\n", sep = "")
cat("Wrote: ", grad_pdf_path, "\n", sep = "")
cat("Wrote: ", probe_pdf_path, "\n", sep = "")
cat("Wrote: ", summary_path, "\n", sep = "")
