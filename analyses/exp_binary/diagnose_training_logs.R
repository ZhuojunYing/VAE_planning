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
seed_arg <- get_arg(9, "1:5")

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
          file_name <- sprintf(
            "lambda_%s_alpha_%s_beta_%s_opportunity_%s_expansion_%s_seed_%d_%dn_training_logs.csv",
            lambda_candidate,
            alpha_candidate,
            beta_candidate,
            opportunity_candidate,
            expansion_decision_version,
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
  NA_character_
}

read_training_log <- function(beta_value, opportunity_value, seed) {
  file_path <- training_log_path(lambda_arg, alpha_arg, beta_value, opportunity_value, seed)
  if (is.na(file_path)) {
    warning(sprintf(
      "Missing training log for beta=%s opportunity=%s seed=%d",
      beta_value,
      opportunity_value,
      seed
    ))
    return(NULL)
  }

  dat <- read.csv(file_path, stringsAsFactors = FALSE)
  dat <- drop_unnamed_index_columns(dat)
  dat$beta <- beta_value
  dat$opportunity <- opportunity_value
  dat$seed <- seed
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
    "expansion_loss", "expansion_stop_rate", "expansion_continue_rate",
    "kl_grad_norm_enc", "kl_grad_norm_lstm", "kl_grad_norm_dec",
    "act_grad_norm_enc", "act_grad_norm_lstm", "act_grad_norm_dec",
    "rec_grad_norm_enc", "rec_grad_norm_lstm", "rec_grad_norm_dec",
    "expansion_epsilon", "expansion_temperature", "expansion_entropy_coef",
    "learning_rate"
  ),
  names(all_logs)
)

for (col in metric_cols) {
  all_logs[[col]] <- suppressWarnings(as.numeric(all_logs[[col]]))
}
all_logs$epoch <- suppressWarnings(as.numeric(all_logs$epoch))

mean_or_na <- function(x) {
  x <- x[is.finite(x)]
  if (length(x) == 0) NA_real_ else mean(x)
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
beta_cols <- palette_for(beta_levels)

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

  for (beta_value in beta_levels) {
    series <- summary_dat[summary_dat$beta == beta_value, , drop = FALSE]
    series <- series[order(series$epoch), , drop = FALSE]
    if (nrow(series) > 0) {
      lines(series$epoch, series$value, col = beta_cols[[beta_value]], lwd = 2)
    }
  }
}

add_beta_legend <- function() {
  old_xpd <- par("xpd")
  par(xpd = NA)
  legend(
    "topright",
    inset = c(-0.35, 0),
    legend = paste("beta", beta_levels),
    col = beta_cols[beta_levels],
    lwd = 2,
    bty = "n"
  )
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

diagnostic_label <- sprintf(
  "lambda_%s_alpha_%s_beta_%s_opportunity_%s_expansion_%s_%dn",
  lambda_arg,
  alpha_arg,
  arg_label(beta_values),
  arg_label(opportunity_values),
  expansion_decision_version,
  tree_size
)

pdf_path <- file.path(results_dir, sprintf("training_diagnostics_%s.pdf", diagnostic_label))
pdf(pdf_path, width = 14, height = 12)
old_par <- par(mfrow = c(4, 3), mar = c(4.5, 4.5, 1, 7))

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

plot_metric(all_logs, "expansion_stop_rate", "Expansion stop rate")
plot_metric(all_logs, "expansion_continue_rate", "Expansion continue rate")
plot_metric(all_logs, "total_loss", "Total loss")
plot_metric(all_logs, "expansion_loss", "Expansion loss")
plot_metric(all_logs, "action_loss", "Action loss")
plot_metric(all_logs, "kl_loss", "KL loss")
plot_metric(all_logs, "reconstruction_loss", "Reconstruction loss")
plot_metric(all_logs, "learning_rate", "Learning rate", log_y = TRUE)
plot_metric(all_logs, "act_to_kl_grad_enc", "Action/KL grad ratio encoder", log_y = TRUE)
plot_metric(all_logs, "act_to_kl_grad_lstm", "Action/KL grad ratio LSTM", log_y = TRUE)
plot_metric(all_logs, "act_to_kl_grad_dec", "Action/KL grad ratio decoder", log_y = TRUE)

add_beta_legend()
par(old_par)
dev.off()

grad_pdf_path <- file.path(results_dir, sprintf("training_gradients_%s.pdf", diagnostic_label))
pdf(grad_pdf_path, width = 14, height = 10)
old_par <- par(mfrow = c(3, 3), mar = c(4.5, 4.5, 1, 7))
plot_metric(all_logs, "act_grad_norm_enc", "Action grad encoder", log_y = TRUE)
plot_metric(all_logs, "act_grad_norm_lstm", "Action grad LSTM", log_y = TRUE)
plot_metric(all_logs, "act_grad_norm_dec", "Action grad decoder", log_y = TRUE)
plot_metric(all_logs, "kl_grad_norm_enc", "KL grad encoder", log_y = TRUE)
plot_metric(all_logs, "kl_grad_norm_lstm", "KL grad LSTM", log_y = TRUE)
plot_metric(all_logs, "kl_grad_norm_dec", "KL grad decoder", log_y = TRUE)
plot_metric(all_logs, "rec_grad_norm_enc", "Recon grad encoder", log_y = TRUE)
plot_metric(all_logs, "rec_grad_norm_lstm", "Recon grad LSTM", log_y = TRUE)
plot_metric(all_logs, "rec_grad_norm_dec", "Recon grad decoder", log_y = TRUE)
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
    final_expansion_stop_rate = if ("expansion_stop_rate" %in% names(dat)) last_row$expansion_stop_rate else NA_real_,
    final_expansion_continue_rate = if ("expansion_continue_rate" %in% names(dat)) last_row$expansion_continue_rate else NA_real_,
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
cat("- Use the action/KL gradient ratio panels to see whether the action signal overwhelms the information signal in the shared encoder/LSTM/decoder.\n")
cat("- Use the first panel to confirm that epsilon was present for enough epochs and did not start/end at zero.\n")
cat("\nWrote: ", pdf_path, "\n", sep = "")
cat("Wrote: ", grad_pdf_path, "\n", sep = "")
cat("Wrote: ", summary_path, "\n", sep = "")
