# ==============================================================================
# KLTN PIPELINE — DỰ ĐOÁN NEUTROPENIA NẶNG
# Thứ tự xử lý:
#   Đọc data → Làm sạch → Split 70/30 → Impute PMM (cột gốc)
#   → dummyVars → make.names → Feature Selection → Modeling
# ==============================================================================

# ==============================================================================
# PHẦN 1: KHAI BÁO THƯ VIỆN
# ==============================================================================
library(readxl);    library(dplyr);     library(tidyr)
library(writexl);   library(mice)
library(caret);     library(pROC);      library(ranger)
library(glmnet);    library(doParallel)
library(xgboost);   library(e1071)
library(ggplot2);   library(gridExtra); library(ggrepel)
library(shapviz);   library(kernelshap)

set.seed(123)
RNGkind(sample.kind = "Rejection")

# ==============================================================================
# PHẦN 2: NHẬP VÀ LÀM SẠCH DỮ LIỆU
# ==============================================================================
df_raw <- read_excel("C:/Users/Hoang Long/Downloads/KLTN_data.xlsx")
df_raw <- df_raw[-1, ]
df_raw <- df_raw[as.numeric(df_raw$Tuổi) >= 18, ]
df_raw <- df_raw %>%
  select(-c(STT, Ma_NC, "Chiều cao", "Cân nặng",
            "out_NEU_cokhong", "out_NEU_muc_do", "out_NEU_FN"))

zero_var_cols <- sapply(df_raw, function(x) length(unique(na.omit(x))) == 1)
cat("Cột bị loại (zero-variance):", names(df_raw)[zero_var_cols], "\n")
df_raw <- df_raw[, !zero_var_cols]

df_raw <- df_raw %>%
  mutate(across(where(is.character), ~ type.convert(.x, as.is = TRUE)))

cat_cols <- c("Loại_ung_thư", "Phác_đồ")
df_raw[cat_cols] <- lapply(df_raw[cat_cols], as.factor)

target_col_orig <- "out_NEU_nang"
numeric_orig    <- c("Tuổi", "BMI", "BSA")
out_cols_orig   <- names(df_raw)[startsWith(names(df_raw), "out_")]

cat("Target gốc:", target_col_orig, "\n")
cat("Numeric gốc:", numeric_orig, "\n")
cat("Outcome cols:", out_cols_orig, "\n")

# ==============================================================================
# PHẦN 3: PHÁT HIỆN NGOẠI LAI
# ==============================================================================
cols_outlier      <- numeric_orig[numeric_orig %in% names(df_raw)]
all_outliers_list <- list()
plot_list_outlier <- list()

for (col_name in cols_outlier) {
  data_vec   <- as.numeric(df_raw[[col_name]])
  out_values <- boxplot.stats(data_vec)$out
  
  plot_df  <- data.frame(value = data_vec, is_outlier = data_vec %in% out_values)
  label_df <- plot_df %>% filter(is_outlier) %>% distinct(value) %>% mutate(x = 1)
  
  p <- ggplot(plot_df, aes(x = 1, y = value)) +
    geom_boxplot(width = 0.4, fill = "#bfdbfe", color = "#1d4ed8",
                 linewidth = 0.8, outlier.shape = NA) +
    geom_jitter(data  = plot_df %>% filter(!is_outlier),
                width = 0.15, alpha = 0.25, size = 1.2, color = "gray60") +
    geom_point(data  = plot_df %>% filter(is_outlier),
               color = "#dc2626", size = 3, shape = 16) +
    ggrepel::geom_label_repel(
      data = label_df, aes(x = x, y = value, label = round(value, 2)),
      color = "#dc2626", fill = "white", size = 3.5, fontface = "bold",
      box.padding = 0.4, point.padding = 0.3,
      segment.color = "#dc2626", segment.size = 0.4,
      min.segment.length = 0, direction = "y",
      nudge_x = 0.35, max.overlaps = Inf) +
    annotate("text", x = 0.72, y = boxplot.stats(data_vec)$stats[5],
             label = paste("↑", round(boxplot.stats(data_vec)$stats[5], 1)),
             size = 3, color = "#1d4ed8", hjust = 1) +
    annotate("text", x = 0.72, y = boxplot.stats(data_vec)$stats[1],
             label = paste("↓", round(boxplot.stats(data_vec)$stats[1], 1)),
             size = 3, color = "#1d4ed8", hjust = 1) +
    labs(title    = col_name,
         subtitle = paste0(length(out_values), " outlier",
                           ifelse(length(out_values) > 1, "s", ""),
                           " | IQR: [",
                           round(quantile(data_vec, 0.25, na.rm = TRUE), 1), "–",
                           round(quantile(data_vec, 0.75, na.rm = TRUE), 1), "]"),
         x = NULL, y = col_name) +
    theme_bw(base_size = 11) +
    theme(axis.text.x = element_blank(), axis.ticks.x = element_blank(),
          plot.title    = element_text(face = "bold", hjust = 0.5),
          plot.subtitle = element_text(size = 8, hjust = 0.5, color = "gray40"),
          panel.grid.major.x = element_blank()) +
    xlim(0.5, 1.8)
  
  plot_list_outlier[[col_name]] <- p
  if (length(out_values) > 0)
    all_outliers_list[[col_name]] <- df_raw[as.numeric(df_raw[[col_name]]) %in% out_values, ]
}

if (length(plot_list_outlier) > 0) {
  ncol_out <- min(3, length(plot_list_outlier))
  nrow_out <- ceiling(length(plot_list_outlier) / ncol_out)
  gridExtra::grid.arrange(
    grobs = plot_list_outlier, nrow = nrow_out, ncol = ncol_out,
    top   = grid::textGrob("Phân phối và Ngoại lai",
                           gp = grid::gpar(fontsize = 14, fontface = "bold")))
}
if (length(all_outliers_list) > 0)
  write_xlsx(all_outliers_list, "Ngoai_lai.xlsx")

# ==============================================================================
# PHẦN 4: TRAIN / TEST SPLIT (70/30, stratified)
# ==============================================================================
set.seed(123)
train_index <- createDataPartition(df_raw[[target_col_orig]], p = 0.7, list = FALSE)
train_raw   <- df_raw[train_index, ]
test_raw    <- df_raw[-train_index, ]
cat(sprintf("Train: %d | Test: %d | Tỷ lệ: %.0f/%.0f\n",
            nrow(train_raw), nrow(test_raw),
            nrow(train_raw) / nrow(df_raw) * 100,
            nrow(test_raw)  / nrow(df_raw) * 100))

# ==============================================================================
# PHẦN 5: IMPUTATION TRÊN CỘT GỐC (trước dummyVars)
# ==============================================================================
cols_to_pmm <- numeric_orig[numeric_orig %in% names(train_raw)]
cols_to_pmm <- cols_to_pmm[sapply(cols_to_pmm,
                                  function(c) any(is.na(as.numeric(train_raw[[c]]))))]
cat("Cột sẽ PMM:", if (length(cols_to_pmm) > 0) cols_to_pmm else "Không có NA\n")

if (length(cols_to_pmm) > 0) {
  cols_for_mice    <- c(numeric_orig[numeric_orig %in% names(train_raw)], target_col_orig)
  cols_for_mice    <- cols_for_mice[cols_for_mice %in% names(train_raw)]
  train_mice_input <- train_raw[, cols_for_mice, drop = FALSE] %>%
    mutate(across(everything(), as.numeric))
  
  init_train              <- mice(train_mice_input, maxit = 0, printFlag = FALSE)
  meth_train              <- init_train$method
  meth_train[cols_to_pmm] <- "pmm"
  
  set.seed(123)
  imputed_obj   <- mice(train_mice_input, method = meth_train,
                        m = 5, maxit = 10, seed = 123, printFlag = FALSE)
  train_imputed <- complete(imputed_obj, 1)
  
  for (col in cols_to_pmm) train_raw[[col]] <- train_imputed[[col]]
  cat("NA còn lại trong train (numeric):", sum(is.na(train_raw[, cols_to_pmm])), "\n")
  
  # Visualisation imputation
  plot_imputation_dist <- function(col_name, before_vec, after_vec, n_miss) {
    df_plot <- bind_rows(
      data.frame(value = before_vec, status = "Trước"),
      data.frame(value = after_vec,  status = "Sau")
    ) %>% filter(!is.na(value))
    df_imp <- data.frame(value = after_vec[which(is.na(before_vec))])
    
    ggplot() +
      geom_density(data = df_plot %>% filter(status == "Trước"),
                   aes(x = value, fill = status, color = status),
                   alpha = 0.4, linewidth = 0.8) +
      geom_density(data = df_plot %>% filter(status == "Sau"),
                   aes(x = value, fill = status, color = status),
                   alpha = 0.4, linewidth = 0.8, linetype = "dashed") +
      geom_rug(data = df_imp, aes(x = value),
               color = "#dc2626", alpha = 0.8, linewidth = 0.6) +
      scale_fill_manual(values  = c("Trước" = "#2563eb", "Sau" = "#16a34a")) +
      scale_color_manual(values = c("Trước" = "#2563eb", "Sau" = "#16a34a")) +
      labs(title    = col_name,
           subtitle = paste0(n_miss, " giá trị được impute (",
                             round(n_miss / length(before_vec) * 100, 1), "%)"),
           x = col_name, y = "Mật độ xác suất",
           fill = NULL, color = NULL, caption = "| = Vị trí điền dữ liệu") +
      theme_bw(base_size = 10) +
      theme(plot.title    = element_text(face = "bold", hjust = 0.5),
            plot.subtitle = element_text(size = 8, hjust = 0.5, color = "gray40"),
            legend.position = "bottom")
  }
  
  dist_plots <- lapply(cols_to_pmm, function(col) {
    before_vec <- as.numeric(train_mice_input[[col]])
    after_vec  <- as.numeric(train_raw[[col]])
    plot_imputation_dist(col, before_vec, after_vec, sum(is.na(before_vec)))
  })
  ncol_imp <- min(2, length(dist_plots))
  gridExtra::grid.arrange(
    grobs = dist_plots, nrow = ceiling(length(dist_plots)/ncol_imp), ncol = ncol_imp,
    top   = grid::textGrob("Phân phối Trước vs Sau Imputation (PMM)",
                           gp = grid::gpar(fontsize = 13, fontface = "bold")))
  
  # Bảng thống kê + KS test
  stats_imp <- lapply(cols_to_pmm, function(col) {
    before <- as.numeric(train_mice_input[[col]])
    after  <- as.numeric(train_raw[[col]])
    data.frame(Variable = col,
               N_missing = sum(is.na(before)), Pct_miss = round(mean(is.na(before))*100,1),
               Mean_before = round(mean(before, na.rm=TRUE),3), Mean_after = round(mean(after),3),
               SD_before   = round(sd(before,   na.rm=TRUE),3), SD_after   = round(sd(after),3),
               Median_before = round(median(before, na.rm=TRUE),3), Median_after = round(median(after),3))
  }) %>% bind_rows()
  cat("\n=== Thống kê trước/sau Imputation ===\n"); print(stats_imp)
  
  ks_results <- lapply(cols_to_pmm, function(col) {
    before <- as.numeric(train_mice_input[[col]])
    after  <- as.numeric(train_raw[[col]])
    ks     <- ks.test(before[!is.na(before)], after)
    data.frame(Variable = col, KS_stat = round(ks$statistic,4), p_value = round(ks$p.value,4),
               Kết_luận = ifelse(ks$p.value > 0.05, "✓ Phân phối ổn định", "⚠ Phân phối thay đổi"))
  }) %>% bind_rows()
  cat("\n=== KS test ===\n"); print(ks_results)
  
  for (col in cols_to_pmm) {
    if (any(is.na(as.numeric(test_raw[[col]])))) {
      fill_val <- median(as.numeric(train_raw[[col]]), na.rm = TRUE)
      test_raw[[col]][is.na(as.numeric(test_raw[[col]]))] <- fill_val
    }
  }
  cat("NA còn lại trong test:", sum(is.na(test_raw[, cols_to_pmm])), "\n")
}

# ==============================================================================
# PHẦN 6: DUMMY ENCODING + MAKE.NAMES
# ==============================================================================
dmy_encoder <- dummyVars(
  formula  = as.formula(paste("~", paste(cat_cols, collapse = " + "))),
  data     = train_raw, fullRank = TRUE)

train_dum    <- as.data.frame(predict(dmy_encoder, newdata = train_raw))
test_dum     <- as.data.frame(predict(dmy_encoder, newdata = test_raw))
non_cat_cols <- setdiff(names(train_raw), cat_cols)
train_df     <- bind_cols(train_raw[, non_cat_cols, drop = FALSE], train_dum)
test_df      <- bind_cols(test_raw[,  non_cat_cols, drop = FALSE], test_dum)

names(train_df) <- make.names(names(train_df), unique = TRUE)
names(test_df)  <- make.names(names(test_df),  unique = TRUE)

target_col   <- make.names(target_col_orig)
numeric_cols <- make.names(numeric_orig)
out_cols     <- make.names(out_cols_orig)

cat("Target column:", target_col, "\n")
cat("Tổng số cột sau dummy:", ncol(train_df), "\n")
stopifnot(identical(names(train_df), names(test_df)))
cat("✓ Train và test cùng cột\n")

# ==============================================================================
# PHẦN 7: CHUẨN BỊ X / Y
# ==============================================================================
prepare_xy <- function(data, target) {
  y <- factor(data[[target]], levels = c(0, 1), labels = c("No", "Yes"))
  drop_cols <- names(data)[startsWith(names(data), "out_")]
  x <- data %>%
    select(-all_of(drop_cols)) %>%
    mutate(across(where(is.factor),    as.numeric)) %>%
    mutate(across(where(is.character), as.numeric))
  list(x = x, y = y)
}

align_cols <- function(x_ref, x_new) {
  for (col in setdiff(names(x_ref), names(x_new))) x_new[[col]] <- NA_real_
  x_new[, names(x_ref), drop = FALSE]
}

train_data <- prepare_xy(train_df, target_col)
test_data  <- prepare_xy(test_df,  target_col)
x_train    <- train_data$x
y_train    <- factor(train_data$y, levels = c("No", "Yes"))
x_test     <- align_cols(x_train, test_data$x)
y_test     <- factor(test_data$y, levels = c("No", "Yes"))

cat(sprintf("✓ x_train: %d cols | x_test: %d cols | Khớp: %s\n",
            ncol(x_train), ncol(x_test), identical(names(x_train), names(x_test))))
cat("Train:", table(y_train), "| Test:", table(y_test), "\n")

# ==============================================================================
# PHẦN 8 (MỞ RỘNG): RFE RIÊNG CHO TỪNG NHÓM MODEL
#
# Mỗi model dùng tập biến tối ưu của riêng mình
# Tiêu chí chọn: ROC không giảm > 1% so với best (parsimony principle)
#
# Lưu ý:
#   • RF   → rfFuncs (caret built-in, ranger wrapper)
#   • XGB  → xgbFuncs (custom, dùng xgb.train trực tiếp)
#   • SVM  → svmFuncs (custom, dùng e1071::svm)
#   • LR   → lrFuncs  (custom, dùng glm/glmnet)
#   • Tất cả fit trên x_train / y_train (TRƯỚC khi tạo down/smote/weighted)
#     → tránh leakage từ sampling strategy vào feature selection
# ==============================================================================

# Hàm helper: chọn final_size theo parsimony (ROC không giảm > 1%)
pick_vars <- function(rfe_res) {
  s        <- rfe_res$results
  best_roc <- max(s$ROC)
  n_vars   <- min(s$Variables[s$ROC >= best_roc - 0.01])
  predictors(rfe_res)[seq_len(n_vars)]
}

# ── 8.1 RFE cho RANDOM FOREST ─────────────────────────────────────────────────
cat("\n[RFE 1/4] Random Forest...\n")
cl <- makeCluster(detectCores() - 1)
registerDoParallel(cl); clusterSetRNGStream(cl, iseed = 123)

rfFuncs_roc         <- rfFuncs
rfFuncs_roc$summary <- twoClassSummary

set.seed(123)
rfe_rf <- rfe(
  x = x_train, y = y_train,
  sizes      = 1:ncol(x_train),
  rfeControl = rfeControl(functions = rfFuncs_roc, method = "cv", number = 10,
                          verbose = FALSE, allowParallel = TRUE),
  metric     = "ROC", ntree = 300,
  strata     = y_train, sampsize = rep(min(table(y_train)), 2))

stopCluster(cl); registerDoSEQ()

vars_rf <- pick_vars(rfe_rf)
cat(sprintf("  RF chọn %d biến (best ROC=%.4f)\n",
            length(vars_rf), max(rfe_rf$results$ROC)))

# ── 8.2 RFE cho XGBOOST (custom wrapper) ──────────────────────────────────────
cat("\n[RFE 2/4] XGBoost...\n")

# Custom RFE functions cho XGBoost
# fit:     train model → trả về object có thể predict
# pred:    predict trên newdata → trả về data.frame(pred, obs, Yes, No)
# rank:    xếp hạng biến theo importance → data.frame(Overall, var)
# selectSize / selectVar: dùng default của caret
xgbFuncs <- list(
  summary = twoClassSummary,
  
  fit = function(x, y, first, last, ...) {
    y_num   <- as.integer(y == "Yes")
    w_yes   <- sum(y=="No") / sum(y=="Yes")   # Weighted để ổn định với imbalance
    xgboost::xgb.train(
      params  = list(objective = "binary:logistic", 
                     eval_metric = "auc",
                     max_depth = 5, 
                     eta = 0.05, 
                     subsample = 0.8,
                     colsample_bytree = 0.8, 
                     min_child_weight = 1,
                     scale_pos_weight = w_yes, 
                     nthread = 1, 
                     verbosity = 0),
      data    = xgboost::xgb.DMatrix(as.matrix(x), label = y_num),
      nrounds = 200, 
      verbose = 0
    )
  },
  
  pred = function(object, x) {
    # Dự đoán xác suất
    prob <- predict(object, xgboost::xgb.DMatrix(as.matrix(x)))
    data.frame(
      Yes  = prob,
      No   = 1 - prob,
      pred = factor(ifelse(prob >= 0.5, "Yes", "No"), levels = c("No", "Yes"))
    )
  },
  
  rank = function(object, x, y) {
    # Tính toán độ quan trọng của biến
    imp <- xgboost::xgb.importance(model = object, feature_names = colnames(x))
    
    all_vars <- data.frame(var = colnames(x))
    
    # Kết hợp dplyr vào để xử lý bảng
    imp_df <- merge(all_vars,
                    imp[, c("Feature", "Gain")],
                    by.x = "var", by.y = "Feature", all.x = TRUE)
    
    imp_df$Gain[is.na(imp_df$Gain)] <- 0
    
    # Trình tự xử lý của dplyr
    imp_df %>%
      dplyr::rename(Overall = Gain) %>%
      dplyr::arrange(desc(Overall))
  },
  
  selectSize = pickSizeBest,
  selectVar  = pickVars
)

cores <- detectCores() - 1
cl <- makeCluster(cores)
registerDoParallel(cl)

# QUAN TRỌNG: Load thư viện cho từng máy con
clusterEvalQ(cl, {
  library(xgboost)
  library(dplyr)
})

# Truyền biến iseed để đảm bảo kết quả có thể tái lập (reproducible)
clusterSetRNGStream(cl, iseed = 123)

set.seed(123)
rfe_xgb <- rfe(
  x = x_train, 
  y = y_train,
  sizes      = 1:ncol(x_train),
  rfeControl = rfeControl(functions = xgbFuncs, 
                          method = "cv", 
                          number = 10,
                          verbose = FALSE, 
                          allowParallel = TRUE,
                          returnResamp = "final"),
  metric     = "ROC"
)

stopCluster(cl); registerDoSEQ()

vars_xgb <- pick_vars(rfe_xgb)
cat(sprintf("  XGB chọn %d biến (best ROC=%.4f)\n",
            length(vars_xgb), max(rfe_xgb$results$ROC)))

# ── 8.3 RFE cho SVM (custom wrapper) ─────────────────────────────────────────
cat("\n[RFE 3/4] SVM...\n")
# Lưu ý: SVM cần dữ liệu scaled → dùng x_train_s (scale toàn bộ x_train)
preproc_full  <- preProcess(x_train, method = c("center","scale"))
x_train_s_full <- predict(preproc_full, x_train)
x_test_s_full  <- predict(preproc_full, x_test)

svmFuncs <- list(
  summary = twoClassSummary,
  
  fit = function(x, y, first, last, ...) {
    w_yes <- sum(y == "No") / sum(y == "Yes")
    # Sử dụng e1071::svm trực tiếp
    e1071::svm(x = as.matrix(x), y = y, kernel = "radial",
               cost = 1, gamma = 0.01,
               probability = TRUE,
               class.weights = c("No" = 1, "Yes" = w_yes))
  },
  
  pred = function(object, x) {
    prob_mat <- attr(
      predict(object, newdata = as.matrix(x), probability = TRUE),
      "probabilities")
    prob_yes <- as.numeric(prob_mat[, "Yes"])
    data.frame(
      Yes  = prob_yes,
      No   = 1 - prob_yes,
      pred = factor(ifelse(prob_yes >= 0.5, "Yes", "No"), levels = c("No", "Yes")))
  },
  
  rank = function(object, x, y) {
    y_num     <- as.integer(y == "Yes")
    prob_base <- attr(predict(object, newdata = as.matrix(x), probability = TRUE),
                      "probabilities")[, "Yes"]
    
    # Chỉ định rõ pROC:: để tránh lỗi find function
    auc_base  <- as.numeric(pROC::auc(pROC::roc(y_num, as.numeric(prob_base), quiet = TRUE)))
    
    imp_vals <- sapply(colnames(x), function(feat) {
      x_perm          <- as.data.frame(x)
      x_perm[[feat]]  <- sample(x_perm[[feat]])
      prob_perm <- attr(
        predict(object, newdata = as.matrix(x_perm), probability = TRUE),
        "probabilities")[, "Yes"]
      
      auc_perm  <- tryCatch(
        as.numeric(pROC::auc(pROC::roc(y_num, as.numeric(prob_perm), quiet = TRUE))),
        error = function(e) auc_base)
      auc_base - auc_perm
    })
    
    # Sử dụng dplyr:: để an toàn trong môi trường song song
    data.frame(var     = colnames(x),
               Overall = pmax(imp_vals, 0)) %>%
      dplyr::arrange(dplyr::desc(Overall))
  },
  
  selectSize = pickSizeBest,
  selectVar  = pickVars
)

# 2. Thiết lập song song và nạp thư viện cho node con
cl <- makeCluster(detectCores() - 1)
registerDoParallel(cl)

# QUAN TRỌNG: Nạp tất cả các package cần thiết vào các core con
clusterEvalQ(cl, {
  library(pROC)
  library(e1071)
  library(dplyr)
})

clusterSetRNGStream(cl, iseed = 123)

# 3. Chạy RFE
set.seed(123)
rfe_svm <- rfe(
  x = x_train_s_full, 
  y = y_train,
  sizes      = 1:ncol(x_train_s_full),
  rfeControl = rfeControl(functions = svmFuncs, 
                          method = "cv", 
                          number = 10,
                          verbose = FALSE, 
                          allowParallel = TRUE,
                          returnResamp = "final"),
  metric     = "ROC"
)

stopCluster(cl); registerDoSEQ()

vars_svm <- pick_vars(rfe_svm)
cat(sprintf("  SVM chọn %d biến (best ROC=%.4f)\n",
            length(vars_svm), max(rfe_svm$results$ROC)))

# ── 8.4 RFE cho LOGISTIC (custom wrapper) ─────────────────────────────────────
cat("\n[RFE 4/4] Logistic...\n")
# LR dùng LASSO (glmnet alpha=1) vì ổn định hơn glm thuần khi nhiều biến
library(glmnet)
library(dplyr)
library(doParallel)

# 1. Định nghĩa lại lrFuncs với chỉ định rõ ràng
lrFuncs <- list(
  summary = twoClassSummary,
  
  fit = function(x, y, first, last, ...) {
    # Sử dụng glmnet:: trực tiếp
    cv_fit <- tryCatch(
      glmnet::cv.glmnet(x = as.matrix(x), y = y, family = "binomial",
                        alpha = 1, nfolds = 5, type.measure = "auc"),
      error = function(e) NULL)
    
    if (is.null(cv_fit)) return(NULL)
    list(model = cv_fit, lambda = cv_fit$lambda.min, col_names = colnames(x))
  },
  
  pred = function(object, x) {
    if (is.null(object)) {
      prob <- rep(0.5, nrow(x))
    } else {
      # Đảm bảo dùng predict của glmnet
      prob <- as.numeric(
        predict(object$model, newx = as.matrix(x[, object$col_names, drop = FALSE]),
                s = object$lambda, type = "response"))
    }
    data.frame(
      Yes  = prob,
      No   = 1 - prob,
      pred = factor(ifelse(prob >= 0.5, "Yes", "No"), levels = c("No", "Yes")))
  },
  
  rank = function(object, x, y) {
    if (is.null(object)) {
      return(data.frame(var = colnames(x), Overall = 0))
    }
    
    # Lấy hệ số (coefficients)
    coef_mat <- coef(object$model, s = object$lambda)
    coef_df  <- data.frame(
      var     = rownames(coef_mat)[-1],
      Overall = abs(as.numeric(coef_mat[-1, 1]))) 
    
    all_vars <- data.frame(var = colnames(x))
    
    # Thay thế %>% bằng cú pháp chuẩn hoặc nạp dplyr đầy đủ
    res <- merge(all_vars, coef_df, by = "var", all.x = TRUE)
    
    # Sử dụng dplyr:: trực tiếp để tránh lỗi "could not find function"
    res <- res %>%
      dplyr::mutate(Overall = ifelse(is.na(Overall), 0, Overall)) %>%
      dplyr::arrange(dplyr::desc(Overall))
    
    return(res)
  },
  
  selectSize = pickSizeBest,
  selectVar  = pickVars
)

# 2. Thiết lập song song
cl <- makeCluster(detectCores() - 1)
registerDoParallel(cl)

# QUAN TRỌNG: Nạp tất cả các package cần thiết vào các core con
clusterEvalQ(cl, {
  library(glmnet)
  library(dplyr)
  library(caret) # Đảm bảo nạp caret để sử dụng twoClassSummary
})

clusterSetRNGStream(cl, iseed = 123)

# 3. Chạy RFE
set.seed(123)
rfe_lr <- rfe(
  x = x_train_s_full, 
  y = y_train,
  sizes      = 1:ncol(x_train_s_full),
  rfeControl = rfeControl(functions = lrFuncs, 
                          method = "cv", 
                          number = 10,
                          verbose = FALSE, 
                          allowParallel = TRUE,
                          returnResamp = "final"),
  metric     = "ROC"
)

stopCluster(cl); registerDoSEQ()

vars_lr <- pick_vars(rfe_lr)
cat(sprintf("  LR chọn %d biến (best ROC=%.4f)\n",
            length(vars_lr), max(rfe_lr$results$ROC)))

# ── 8.5 Tổng hợp và visualisation ─────────────────────────────────────────────
rfe_summary_all <- data.frame(
  Model    = c("Random Forest","XGBoost","SVM","Logistic"),
  N_vars   = c(length(vars_rf), length(vars_xgb),
               length(vars_svm), length(vars_lr)),
  Best_AUC = c(round(max(rfe_rf$results$ROC),  4),
               round(max(rfe_xgb$results$ROC), 4),
               round(max(rfe_svm$results$ROC), 4),
               round(max(rfe_lr$results$ROC),  4)))
cat("\n=== RFE Summary ===\n"); print(rfe_summary_all)

# Plot RFE — 4 panels
plot_rfe <- function(rfe_res, title, n_chosen) {
  s <- rfe_res$results
  ggplot(s, aes(x=Variables, y=ROC)) +
    geom_ribbon(aes(ymin=ROC-ROCSD, ymax=ROC+ROCSD), alpha=0.15, fill="#2563eb") +
    geom_line(color="#2563eb", linewidth=1) + geom_point(color="#2563eb", size=2) +
    geom_vline(xintercept=n_chosen, linetype="dashed", color="red") +
    annotate("text", x=n_chosen+0.5, y=min(s$ROC),
             label=paste("Chọn:", n_chosen, "biến"), color="red", hjust=0, size=3) +
    labs(title=title, x="Số biến", y="AUC (CV)") +
    theme_bw(base_size=10) +
    theme(plot.title=element_text(face="bold", hjust=0.5))
}

gridExtra::grid.arrange(
  plot_rfe(rfe_rf,  "RFE — Random Forest", length(vars_rf)),
  plot_rfe(rfe_xgb, "RFE — XGBoost",       length(vars_xgb)),
  plot_rfe(rfe_svm, "RFE — SVM",           length(vars_svm)),
  plot_rfe(rfe_lr,  "RFE — Logistic",       length(vars_lr)),
  nrow=2, ncol=2,
  top=grid::textGrob("RFE: Số biến vs AUC (10-fold CV) — 4 nhóm model",
                     gp=grid::gpar(fontsize=13, fontface="bold")))

# Biến chung và biến riêng
all_var_sets <- list(RF=vars_rf, XGB=vars_xgb, SVM=vars_svm, LR=vars_lr)
vars_common  <- Reduce(intersect, all_var_sets)
cat(sprintf("\nBiến chung cả 4 model (%d):\n", length(vars_common)))
print(vars_common)

# ==============================================================================
# CHUẨN BỊ X_TRAIN / X_TEST RIÊNG CHO TỪNG NHÓM MODEL
#
# RF và XGB: dùng unscaled data với vars_rf / vars_xgb
# SVM và LR: dùng scaled data với vars_svm / vars_lr
#   → scale fit trên x_train, transform x_test (tránh leakage)
# ==============================================================================

# RF feature set
x_train_rf <- x_train[, vars_rf, drop=FALSE]
x_test_rf  <- x_test[,  vars_rf, drop=FALSE]

# XGB feature set (unscaled)
x_train_xgb <- x_train[, vars_xgb, drop=FALSE]
x_test_xgb  <- x_test[,  vars_xgb, drop=FALSE]

# SVM feature set (scaled riêng theo vars_svm)
preproc_svm   <- preProcess(x_train[, vars_svm, drop=FALSE], method=c("center","scale"))
x_train_svm   <- predict(preproc_svm, x_train[, vars_svm, drop=FALSE])
x_test_svm    <- predict(preproc_svm, x_test[,  vars_svm, drop=FALSE])

# LR feature set (scaled riêng theo vars_lr)
preproc_lr    <- preProcess(x_train[, vars_lr, drop=FALSE], method=c("center","scale"))
x_train_lr    <- predict(preproc_lr, x_train[, vars_lr, drop=FALSE])
x_test_lr     <- predict(preproc_lr, x_test[,  vars_lr, drop=FALSE])

# Loại cột tương quan cao cho LR (bảo vệ glm)
cor_mat_lr  <- cor(as.matrix(x_train_lr), use="complete.obs")
drop_cor_lr <- findCorrelation(cor_mat_lr, cutoff=0.95, verbose=FALSE)
if (length(drop_cor_lr) > 0) {
  cat(sprintf("► LR: loại %d cột (r>0.95)\n", length(drop_cor_lr)))
  x_train_lr <- x_train_lr[, -drop_cor_lr, drop=FALSE]
  x_test_lr  <- x_test_lr[,  names(x_train_lr), drop=FALSE]
}

cat(sprintf("\nFeature sets:\n"))
cat(sprintf("  RF:  %d train | %d test\n", ncol(x_train_rf),  ncol(x_test_rf)))
cat(sprintf("  XGB: %d train | %d test\n", ncol(x_train_xgb), ncol(x_test_xgb)))
cat(sprintf("  SVM: %d train | %d test\n", ncol(x_train_svm), ncol(x_test_svm)))
cat(sprintf("  LR:  %d train | %d test\n", ncol(x_train_lr),  ncol(x_test_lr)))

# ==============================================================================
# PHẦN 9: IMBALANCE + SAMPLING
# Tạo dataset down/weighted riêng cho từng nhóm (theo feature set riêng)
# ==============================================================================
cat("\n=== IMBALANCE ===\n")
n_minority      <- sum(y_train == "Yes")
n_majority      <- sum(y_train == "No")
weight_yes      <- n_majority / n_minority
sample_weights  <- ifelse(y_train == "Yes", weight_yes, 1)
cat(sprintf("Minority: %d | Majority: %d | Ratio: %.1f:1\n",
            n_minority, n_majority, weight_yes))

# Down 3:1 index (chung — chỉ phụ thuộc y_train, không phụ thuộc feature)
set.seed(123)
n_keep          <- min(n_minority * 3, n_majority)
down_idx        <- c(sample(which(y_train=="No"), n_keep), which(y_train=="Yes"))
y_tr_down3      <- y_train[down_idx]

# Dataset theo từng nhóm
# RF
x_tr_rf_none   <- x_train_rf;          y_tr_rf_none   <- y_train
x_tr_rf_down3  <- x_train_rf[down_idx,]; y_tr_rf_down3  <- y_tr_down3

# XGB
x_tr_xgb_none  <- x_train_xgb;           y_tr_xgb_none  <- y_train
x_tr_xgb_down3 <- x_train_xgb[down_idx,]; y_tr_xgb_down3 <- y_tr_down3

# SVM
x_tr_svm_none  <- x_train_svm;           y_tr_svm_none  <- y_train
x_tr_svm_down3 <- x_train_svm[down_idx,]; y_tr_svm_down3 <- y_tr_down3

# LR (scaled)
x_tr_lr_none   <- x_train_lr; y_tr_lr_none   <- y_train

# Fold index
set.seed(123); idx_rf_none    <- createFolds(y_tr_rf_none,   k=10, returnTrain=TRUE)
set.seed(123); idx_rf_down3   <- createFolds(y_tr_rf_down3,  k=10, returnTrain=TRUE)
set.seed(123); idx_xgb_none   <- createFolds(y_tr_xgb_none,  k=10, returnTrain=TRUE)
set.seed(123); idx_xgb_down3  <- createFolds(y_tr_xgb_down3, k=10, returnTrain=TRUE)
set.seed(123); idx_svm_none   <- createFolds(y_tr_svm_none,  k=10, returnTrain=TRUE)
set.seed(123); idx_svm_down3  <- createFolds(y_tr_svm_down3, k=10, returnTrain=TRUE)
set.seed(123); idx_lr_none    <- createFolds(y_tr_lr_none,   k=10, returnTrain=TRUE)
set.seed(123); idx_fs         <- createFolds(y_train,         k=10, returnTrain=TRUE)

# ==============================================================================
# ==============================================================================
# PHẦN 10: HUẤN LUYỆN — mỗi nhóm dùng feature set RFE riêng
# ==============================================================================

# ── NHÓM 1: RANDOM FOREST ─────────────────────────────────────────────────────
cl <- makeCluster(detectCores()-1)
registerDoParallel(cl); clusterSetRNGStream(cl, iseed=123)

make_ctrl <- function(index = NULL) {
  trainControl(method = "cv", number = 10, index = index,
               classProbs = TRUE, summaryFunction = twoClassSummary,
               savePredictions = "final", verboseIter = FALSE,
               allowParallel = TRUE)
}
make_ctrl_smote <- function(index = NULL) {
  trainControl(method = "cv", number = 10, index = index,
               classProbs = TRUE, summaryFunction = twoClassSummary,
               savePredictions = "final", verboseIter = FALSE,
               allowParallel = TRUE, sampling = "smote")
}
tuning_grid_rf <- expand.grid(
  mtry          = as.integer(round(seq(2, max(2, floor(ncol(x_train_rf)*0.8)),
                                       length.out=5))),
  splitrule     = "extratrees",
  min.node.size = c(1,3,5,10)) %>% unique()

set.seed(123)
rf_none <- train(x=x_tr_rf_none, y=y_tr_rf_none, method="ranger",
                 trControl=make_ctrl(idx_rf_none), tuneGrid=tuning_grid_rf,
                 metric="ROC", importance="permutation", num.trees=500)

set.seed(123)
rf_down3 <- train(x=x_tr_rf_down3, y=y_tr_rf_down3, method="ranger",
                  trControl=make_ctrl(idx_rf_down3), tuneGrid=tuning_grid_rf,
                  metric="ROC", importance="permutation", num.trees=500)

set.seed(123)
rf_smote <- train(x=x_train_rf, y=y_train, method="ranger",
                  trControl=make_ctrl_smote(idx_fs), tuneGrid=tuning_grid_rf,
                  metric="ROC", importance="permutation", num.trees=500)

set.seed(123)
rf_weighted <- train(x=x_tr_rf_none, y=y_tr_rf_none, method="ranger",
                     trControl=make_ctrl(idx_rf_none), tuneGrid=tuning_grid_rf,
                     metric="ROC", importance="permutation", num.trees=500,
                     weights=sample_weights)

stopCluster(cl); registerDoSEQ()

# ── NHÓM 2: XGBOOST ───────────────────────────────────────────────────────────
tuning_grid_xgb <- expand.grid(
  nrounds=c(100,200,300), max_depth=c(3,5,7), eta=c(0.01,0.05,0.1),
  gamma=0, colsample_bytree=c(0.6,0.8), min_child_weight=c(1,5), subsample=0.8)

tune_xgb <- function(x_tr, y_tr, fold_idx, grid, scale_pos_w = 1, label = "XGB") {
  y_num <- as.integer(y_tr == "Yes")
  
  grid_results <- lapply(seq_len(nrow(grid)), function(g) {
    params <- list(
      objective        = "binary:logistic",
      eval_metric      = "auc",
      max_depth        = grid$max_depth[g],
      eta              = grid$eta[g],
      gamma            = grid$gamma[g],
      colsample_bytree = grid$colsample_bytree[g],
      min_child_weight = grid$min_child_weight[g],
      subsample        = grid$subsample[g],
      scale_pos_weight = scale_pos_w,
      nthread          = 1, verbosity = 0)
    
    auc_folds <- sapply(seq_along(fold_idx), function(i) {
      tr_idx  <- fold_idx[[i]]
      val_idx <- setdiff(seq_len(nrow(x_tr)), tr_idx)
      if (length(unique(y_num[val_idx])) < 2) return(NA_real_)
      fit  <- xgb.train(params = params, verbose = 0, nrounds = grid$nrounds[g],
                        data = xgb.DMatrix(as.matrix(x_tr[tr_idx,]), label = y_num[tr_idx]))
      prob <- predict(fit, xgb.DMatrix(as.matrix(x_tr[val_idx,])))
      as.numeric(auc(roc(y_num[val_idx], prob, quiet = TRUE)))
    })
    
    data.frame(grid[g,], AUC_CV = round(mean(auc_folds, na.rm=TRUE),4),
               AUC_SD = round(sd(auc_folds, na.rm=TRUE),4))
  }) %>% bind_rows()
  
  best     <- grid_results[which.max(grid_results$AUC_CV), ]
  cat(sprintf("  %s | AUC CV: %.4f ± %.4f | nrounds=%d, depth=%d, eta=%.3f\n",
              label, best$AUC_CV, best$AUC_SD, best$nrounds, best$max_depth, best$eta))
  
  final_model <- xgb.train(
    params  = list(objective="binary:logistic", eval_metric="auc",
                   max_depth=best$max_depth, eta=best$eta, gamma=best$gamma,
                   colsample_bytree=best$colsample_bytree,
                   min_child_weight=best$min_child_weight, subsample=best$subsample,
                   scale_pos_weight=scale_pos_w, nthread=1, verbosity=0),
    data    = xgb.DMatrix(as.matrix(x_tr), label=y_num),
    nrounds = best$nrounds, verbose = 0)
  
  list(model=final_model, best_params=best, grid_results=grid_results,
       col_names=colnames(x_tr), label=label)
}

get_prob_xgb <- function(xgb_obj, x_new) {
  predict(xgb_obj$model,
          xgb.DMatrix(as.matrix(x_new[, xgb_obj$col_names, drop=FALSE])))
}

set.seed(123); xgb_none_obj <- tune_xgb(
  x_tr_xgb_none,  y_tr_xgb_none,  idx_xgb_none,  tuning_grid_xgb, label="XGB None")
set.seed(123); xgb_down3_obj <- tune_xgb(
  x_tr_xgb_down3, y_tr_xgb_down3, idx_xgb_down3, tuning_grid_xgb, label="XGB Down")
set.seed(123); xgb_smote_obj <- tune_xgb(
  x_train_xgb,    y_train,         idx_fs,         tuning_grid_xgb, label="XGB SMOTE")
set.seed(123); xgb_weighted_obj <- tune_xgb(
  x_tr_xgb_none,  y_tr_xgb_none,  idx_xgb_none,  tuning_grid_xgb,
  scale_pos_w=weight_yes, label="XGB Weighted")

xgb_tuning_summary <- bind_rows(
  xgb_none_obj$best_params     %>% mutate(Model="XGB None"),
  xgb_down3_obj$best_params    %>% mutate(Model="XGB Down"),
  xgb_smote_obj$best_params    %>% mutate(Model="XGB SMOTE"),
  xgb_weighted_obj$best_params %>% mutate(Model="XGB Weighted")
) %>% select(Model, nrounds, max_depth, eta, colsample_bytree, min_child_weight, AUC_CV, AUC_SD)

# ── NHÓM 3: SVM ───────────────────────────────────────────────────────────────
tuning_grid_svm <- expand.grid(C=c(0.1,0.5,1,5,10), sigma=c(0.001,0.005,0.01,0.05,0.1))

tune_svm_e1071 <- function(x_tr, y_tr, fold_idx, grid, class_w=NULL, label="SVM") {
  grid_results <- lapply(seq_len(nrow(grid)), function(g) {
    auc_folds <- sapply(seq_along(fold_idx), function(i) {
      tr_idx  <- fold_idx[[i]]
      val_idx <- setdiff(seq_len(nrow(x_tr)), tr_idx)
      if (length(unique(as.character(y_tr[val_idx]))) < 2) return(NA_real_)
      
      fit <- e1071::svm(x=as.matrix(x_tr[tr_idx,]), y=y_tr[tr_idx],
                        kernel="radial", cost=grid$C[g], gamma=grid$sigma[g],
                        probability=TRUE, class.weights=class_w)
      
      prob_mat <- attr(predict(fit, newdata=as.matrix(x_tr[val_idx,]),
                               probability=TRUE), "probabilities")
      if (!"Yes" %in% colnames(prob_mat)) return(NA_real_)
      as.numeric(auc(roc(y_tr[val_idx], as.numeric(prob_mat[,"Yes"]), quiet=TRUE)))
    })
    
    data.frame(C=grid$C[g], sigma=grid$sigma[g],
               AUC_CV=round(mean(auc_folds,na.rm=TRUE),4),
               AUC_SD=round(sd(auc_folds,  na.rm=TRUE),4))
  }) %>% bind_rows()
  
  best <- grid_results[which.max(grid_results$AUC_CV), ]
  cat(sprintf("  %s | AUC CV: %.4f ± %.4f | C=%.1f, sigma=%.4f\n",
              label, best$AUC_CV, best$AUC_SD, best$C, best$sigma))
  
  final_model <- e1071::svm(x=as.matrix(x_tr), y=y_tr, kernel="radial",
                            cost=best$C, gamma=best$sigma,
                            probability=TRUE, class.weights=class_w)
  
  list(model=final_model, best_params=best, grid_results=grid_results,
       col_names=colnames(x_tr), label=label)
}

get_prob_svm <- function(svm_obj, x_new) {
  prob_mat <- attr(
    predict(svm_obj$model,
            newdata=as.matrix(x_new[, svm_obj$col_names, drop=FALSE]),
            probability=TRUE), "probabilities")
  as.numeric(prob_mat[, "Yes"])
}

set.seed(123); svm_none_obj <- tune_svm_e1071(
  x_tr_svm_none,  y_tr_svm_none,  idx_svm_none,  tuning_grid_svm, label="SVM None")
set.seed(123); svm_down3_obj <- tune_svm_e1071(
  x_tr_svm_down3, y_tr_svm_down3, idx_svm_down3, tuning_grid_svm, label="SVM Down")
set.seed(123); svm_smote_obj <- tune_svm_e1071(
  x_train_svm,    y_train,         idx_fs,         tuning_grid_svm, label="SVM SMOTE")
set.seed(123); svm_weighted_obj <- tune_svm_e1071(
  x_tr_svm_none,  y_tr_svm_none,  idx_svm_none,  tuning_grid_svm,
  class_w=c("No"=1,"Yes"=weight_yes), label="SVM Weighted")

get_prob_svm <- function(svm_obj, x_new) {
  prob_mat <- attr(
    predict(svm_obj$model,
            newdata=as.matrix(x_new[, svm_obj$col_names, drop=FALSE]),
            probability=TRUE), "probabilities")
  as.numeric(prob_mat[, "Yes"])
}

svm_tuning_summary <- bind_rows(
  svm_none_obj$best_params     %>% mutate(Model = "SVM None"),
  svm_down3_obj$best_params    %>% mutate(Model = "SVM Down"),
  svm_smote_obj$best_params    %>% mutate(Model = "SVM SMOTE"),
  svm_weighted_obj$best_params %>% mutate(Model = "SVM Weighted")
) %>% select(Model, C, sigma, AUC_CV, AUC_SD)

# ── NHÓM 4: LOGISTIC ──────────────────────────────────────────────────────────
cl <- makeCluster(detectCores()-1)
registerDoParallel(cl); clusterSetRNGStream(cl, iseed=123)

set.seed(123)
lr_model <- tryCatch({
  train(x=x_tr_lr_none, y=y_tr_lr_none, method="glm", family="binomial",
        trControl=make_ctrl(idx_lr_none), metric="ROC")
}, error=function(e) {
  train(x=x_tr_lr_none, y=y_tr_lr_none, method="glmnet",
        trControl=make_ctrl(idx_lr_none),
        tuneGrid=expand.grid(alpha=0, lambda=10^seq(-3,1,length.out=30)),
        metric="ROC")
})

set.seed(123)
lr_lasso <- train(x=x_tr_lr_none, y=y_tr_lr_none, method="glmnet",
                  trControl=make_ctrl(idx_lr_none),
                  tuneGrid=expand.grid(alpha=1, lambda=10^seq(-4,1,length.out=50)),
                  metric="ROC")

stopCluster(cl); registerDoSEQ()
# ==============================================================================
# PHẦN 10.5: PLATT SCALING + OOF CALIBRATION
# Vị trí: SAU khi train xong tất cả final models (Phần 10)
#         TRƯỚC khi evaluate trên test set (Phần 13)
#
# Quy trình đúng (không leakage):
#   1. Dùng CÙNG fold index đã dùng khi train (idx_rf_none, idx_xgb_none, ...)
#      → OOF scores được tạo bởi model chưa thấy quan sát đó
#      → Giống điều kiện thực tế khi dùng model trên bệnh nhân mới
#   2. Fit Platt scaler trên OOF scores của TRAIN set
#   3. Transform TEST set scores qua scaler đã fit
#      → Test set không tham gia vào bất kỳ bước training nào
#
# Tại sao OOF thay vì in-sample:
#   In-sample scores quá "sạch" (model đã thấy dữ liệu này khi train)
#   → Platt scaler fit trên đó sẽ overfit
#   → Khi apply lên test, calibration bị lệch
# ==============================================================================

# Hàm tạo OOF scores cho RF (caret object)
get_oof_scores_rf <- function(model, x_tr, y_tr, fold_idx) {
  # Dùng savePredictions = "final" đã có sẵn trong trainControl
  # → caret đã lưu OOF predictions trong model$pred
  if (!is.null(model$pred)) {
    # Lấy OOF predictions tại bestTune
    oof_df <- model$pred
    # Filter đúng hyperparameter được chọn
    for (param in names(model$bestTune)) {
      oof_df <- oof_df[oof_df[[param]] == model$bestTune[[param]], ]
    }
    # Sắp xếp theo rowIndex để match với y_tr
    oof_df  <- oof_df[order(oof_df$rowIndex), ]
    scores  <- oof_df$Yes
    indices <- oof_df$rowIndex
    return(list(scores = scores, indices = indices))
  }
  return(NULL)
}

# Hàm tạo OOF scores cho XGBoost (manual)
get_oof_scores_xgb <- function(xgb_obj, x_tr, y_tr, fold_idx) {
  y_num  <- as.integer(y_tr == "Yes")
  scores  <- numeric(nrow(x_tr))
  indices <- integer(nrow(x_tr))
  
  for (i in seq_along(fold_idx)) {
    tr_idx  <- fold_idx[[i]]
    val_idx <- setdiff(seq_len(nrow(x_tr)), tr_idx)
    
    # Train model tạm trên fold này với CÙNG best params
    bp  <- xgb_obj$best_params
    fit <- xgb.train(
      params  = list(objective="binary:logistic", eval_metric="auc",
                     max_depth=bp$max_depth, eta=bp$eta, gamma=bp$gamma,
                     colsample_bytree=bp$colsample_bytree,
                     min_child_weight=bp$min_child_weight,
                     subsample=bp$subsample, nthread=1, verbosity=0),
      data    = xgb.DMatrix(as.matrix(x_tr[tr_idx,]), label=y_num[tr_idx]),
      nrounds = bp$nrounds, verbose=0)
    
    scores[val_idx]  <- predict(fit, xgb.DMatrix(as.matrix(x_tr[val_idx,])))
    indices[val_idx] <- val_idx
  }
  list(scores=scores, indices=seq_len(nrow(x_tr)))
}

# Hàm tạo OOF scores cho SVM (e1071 manual)
get_oof_scores_svm <- function(svm_obj, x_tr, y_tr, fold_idx) {
  scores  <- numeric(nrow(x_tr))
  
  for (i in seq_along(fold_idx)) {
    tr_idx  <- fold_idx[[i]]
    val_idx <- setdiff(seq_len(nrow(x_tr)), tr_idx)
    
    fit <- e1071::svm(
      x=as.matrix(x_tr[tr_idx,]), y=y_tr[tr_idx],
      kernel="radial",
      cost=svm_obj$best_params$C,
      gamma=svm_obj$best_params$sigma,
      probability=TRUE,
      class.weights=if(!is.null(svm_obj$model$class.weights))
        svm_obj$model$class.weights else NULL)
    
    prob_mat <- attr(
      predict(fit, newdata=as.matrix(x_tr[val_idx,]), probability=TRUE),
      "probabilities")
    scores[val_idx] <- as.numeric(prob_mat[,"Yes"])
  }
  list(scores=scores, indices=seq_len(nrow(x_tr)))
}

# Hàm fit Platt scaler từ OOF scores
# Platt scaling: fit logistic regression: log-odds(y) ~ score
fit_platt <- function(oof_scores, y_tr, indices=NULL) {
  if (!is.null(indices))
    y_cal <- as.integer(y_tr[indices] == "Yes")
  else
    y_cal <- as.integer(y_tr == "Yes")
  
  # Clip scores để tránh log(0)
  oof_clipped <- pmax(pmin(oof_scores, 1 - 1e-7), 1e-7)
  
  # Fit logistic regression trên OOF scores
  cal_df  <- data.frame(score = oof_clipped, y = y_cal)
  platt   <- glm(y ~ score, data=cal_df, family=binomial())
  platt
}

# Hàm apply Platt scaler lên scores mới
apply_platt <- function(platt_model, raw_scores) {
  scores_clipped <- pmax(pmin(raw_scores, 1-1e-7), 1e-7)
  as.numeric(predict(platt_model,
                     newdata=data.frame(score=scores_clipped),
                     type="response"))
}

cat("\n=== PLATT SCALING + OOF CALIBRATION ===\n")

# ── RF: OOF từ model$pred (caret đã lưu sẵn) ─────────────────────────────────
cat("Fitting Platt scalers...\n")

# RF None — đại diện cho cả nhóm RF (dùng idx_rf_none)
oof_rf_none <- get_oof_scores_rf(rf_none, x_tr_rf_none, y_tr_rf_none, idx_rf_none)
oof_rf_down3 <- get_oof_scores_rf(rf_down3, x_tr_rf_down3, y_tr_rf_down3, idx_rf_down3)
oof_rf_smote <- get_oof_scores_rf(rf_smote, x_train_rf, y_train, idx_fs)
oof_rf_weighted <- get_oof_scores_rf(rf_weighted, x_tr_rf_none, y_tr_rf_none, idx_rf_none)

platt_rf_none    <- fit_platt(oof_rf_none$scores,    y_tr_rf_none,    oof_rf_none$indices)
platt_rf_down3   <- fit_platt(oof_rf_down3$scores,   y_tr_rf_down3,   oof_rf_down3$indices)
platt_rf_smote   <- fit_platt(oof_rf_smote$scores,   y_train,         oof_rf_smote$indices)
platt_rf_weighted <- fit_platt(oof_rf_weighted$scores, y_tr_rf_none,  oof_rf_weighted$indices)
cat("  ✓ RF Platt scalers fitted\n")

# ── XGBoost: OOF thủ công ────────────────────────────────────────────────────
oof_xgb_none    <- get_oof_scores_xgb(xgb_none_obj,    x_tr_xgb_none,  y_tr_xgb_none,  idx_xgb_none)
oof_xgb_down3   <- get_oof_scores_xgb(xgb_down3_obj,   x_tr_xgb_down3, y_tr_xgb_down3, idx_xgb_down3)
oof_xgb_smote   <- get_oof_scores_xgb(xgb_smote_obj,   x_train_xgb,    y_train,         idx_fs)
oof_xgb_weighted <- get_oof_scores_xgb(xgb_weighted_obj, x_tr_xgb_none, y_tr_xgb_none, idx_xgb_none)

platt_xgb_none    <- fit_platt(oof_xgb_none$scores,    y_tr_xgb_none)
platt_xgb_down3   <- fit_platt(oof_xgb_down3$scores,   y_tr_xgb_down3)
platt_xgb_smote   <- fit_platt(oof_xgb_smote$scores,   y_train)
platt_xgb_weighted <- fit_platt(oof_xgb_weighted$scores, y_tr_xgb_none)
cat("  ✓ XGB Platt scalers fitted\n")

# ── SVM: OOF thủ công ────────────────────────────────────────────────────────
oof_svm_none    <- get_oof_scores_svm(svm_none_obj,    x_tr_svm_none,  y_tr_svm_none,  idx_svm_none)
oof_svm_down3   <- get_oof_scores_svm(svm_down3_obj,   x_tr_svm_down3, y_tr_svm_down3, idx_svm_down3)
oof_svm_smote   <- get_oof_scores_svm(svm_smote_obj,   x_train_svm,    y_train,         idx_fs)
oof_svm_weighted <- get_oof_scores_svm(svm_weighted_obj, x_tr_svm_none, y_tr_svm_none, idx_svm_none)

platt_svm_none    <- fit_platt(oof_svm_none$scores,    y_tr_svm_none)
platt_svm_down3   <- fit_platt(oof_svm_down3$scores,   y_tr_svm_down3)
platt_svm_smote   <- fit_platt(oof_svm_smote$scores,   y_train)
platt_svm_weighted <- fit_platt(oof_svm_weighted$scores, y_tr_svm_none)
cat("  ✓ SVM Platt scalers fitted\n")

# ── LR: OOF từ model$pred ────────────────────────────────────────────────────
oof_lr_model <- get_oof_scores_rf(lr_model, x_tr_lr_none, y_tr_lr_none, idx_lr_none)
oof_lr_lasso <- get_oof_scores_rf(lr_lasso, x_tr_lr_none, y_tr_lr_none, idx_lr_none)

platt_lr_model <- fit_platt(oof_lr_model$scores, y_tr_lr_none, oof_lr_model$indices)
platt_lr_lasso <- fit_platt(oof_lr_lasso$scores, y_tr_lr_none, oof_lr_lasso$indices)
cat("  ✓ LR Platt scalers fitted\n")


# ==============================================================================
# PHẦN 11: KẾT QUẢ HYPERPARAMETER TUNING
# FIX: Tách riêng bảng caret (RF/LR) và manual (XGB/SVM) vì khác cấu trúc
# ==============================================================================

# Bảng RF + LR (caret object — có $results)
caret_model_list  <- list(rf_none, rf_down3, rf_smote, rf_weighted, lr_model, lr_lasso)
caret_model_names <- c("RF None","RF Down","RF SMOTE","RF Weighted","Logistic","Logistic LASSO")

summary_tuning_caret <- lapply(seq_along(caret_model_list), function(i) {
  m        <- caret_model_list[[i]]
  best_row <- merge(m$results, m$bestTune)
  data.frame(Model   = caret_model_names[[i]],
             AUC_CV  = round(best_row$ROC[1],  4),
             Sens_CV = round(best_row$Sens[1], 4),
             Spec_CV = round(best_row$Spec[1], 4),
             stringsAsFactors = FALSE)
}) %>% bind_rows()

cat("\n=== RF + LR Tuning (xếp theo AUC_CV) ===\n")
print(summary_tuning_caret %>% arrange(desc(AUC_CV)))
cat("\n=== XGBoost Tuning ===\n")
print(xgb_tuning_summary   %>% arrange(desc(AUC_CV)))
cat("\n=== SVM Tuning ===\n")
print(svm_tuning_summary   %>% arrange(desc(AUC_CV)))

# ── Plot RF tuning ────────────────────────────────────────────────────────────
plot_tuning_rf <- function(model, title) {
  res <- model$results %>% mutate(min.node.size = factor(min.node.size))
  ggplot(res, aes(x=mtry, y=ROC, color=min.node.size, group=min.node.size)) +
    geom_line(linewidth=0.8) + geom_point(size=2.5) +
    geom_errorbar(aes(ymin=ROC-ROCSD, ymax=ROC+ROCSD), width=0.3, alpha=0.4) +
    geom_point(data=merge(res, model$bestTune), color="red", size=4, shape=8) +
    scale_color_brewer(palette="Set2", name="min.node.size") +
    labs(title=title,
         subtitle=paste0("★ Best: mtry=", model$bestTune$mtry,
                         ", node=", model$bestTune$min.node.size,
                         " | AUC=", round(max(res$ROC),4)),
         x="mtry", y="AUC (CV)") +
    theme_bw(base_size=10) +
    theme(plot.title=element_text(face="bold", hjust=0.5),
          plot.subtitle=element_text(size=8, color="gray40", hjust=0.5))
}

gridExtra::grid.arrange(
  grobs = list(plot_tuning_rf(rf_none,"RF None"), plot_tuning_rf(rf_down3,"RF Down"),
               plot_tuning_rf(rf_smote,"RF SMOTE"), plot_tuning_rf(rf_weighted,"RF Weighted")),
  nrow=2, ncol=2,
  top=grid::textGrob("Tuning — Random Forest", gp=grid::gpar(fontsize=13, fontface="bold")))

# ── Plot XGBoost tuning (từ grid_results của manual object) ──────────────────
# FIX: Dùng xgb_obj$grid_results thay vì model$results (không tồn tại)
plot_xgb_grid <- function(xgb_obj) {
  res      <- xgb_obj$grid_results %>%
    mutate(max_depth=factor(max_depth), eta=factor(eta))
  best_row <- xgb_obj$best_params
  
  ggplot(res, aes(x=nrounds, y=AUC_CV, color=max_depth,
                  group=interaction(max_depth, eta))) +
    geom_line(aes(linetype=eta), linewidth=0.7, alpha=0.8) +
    geom_point(data = best_row %>%
                 mutate(max_depth=factor(max_depth), eta=factor(eta)),
               aes(x=nrounds, y=AUC_CV),
               color="red", size=4, shape=8, inherit.aes=FALSE) +
    scale_color_brewer(palette="Set1", name="max_depth") +
    labs(title    = xgb_obj$label,
         subtitle = sprintf("★ nrounds=%d, depth=%d, eta=%.2f | AUC=%.4f",
                            best_row$nrounds, best_row$max_depth,
                            best_row$eta, best_row$AUC_CV),
         x="nrounds", y="AUC (CV)") +
    theme_bw(base_size=10) +
    theme(plot.title=element_text(face="bold", hjust=0.5),
          plot.subtitle=element_text(size=7, color="gray40", hjust=0.5))
}

gridExtra::grid.arrange(
  grobs = list(plot_xgb_grid(xgb_none_obj), plot_xgb_grid(xgb_down3_obj),
               plot_xgb_grid(xgb_smote_obj), plot_xgb_grid(xgb_weighted_obj)),
  nrow=2, ncol=2,
  top=grid::textGrob("Tuning — XGBoost (Manual CV)", gp=grid::gpar(fontsize=13, fontface="bold")))

# ── Plot SVM tuning (heatmap C × sigma từ svm_obj$grid_results) ──────────────
# FIX: Dùng svm_obj$grid_results thay vì model$results
plot_svm_grid <- function(svm_obj) {
  res      <- svm_obj$grid_results
  best_row <- svm_obj$best_params
  
  ggplot(res, aes(x=factor(sigma), y=factor(C), fill=AUC_CV)) +
    geom_tile(color="white") +
    geom_text(aes(label=round(AUC_CV,3)), size=2.8) +
    geom_tile(data=data.frame(sigma=factor(best_row$sigma), C=factor(best_row$C)),
              aes(x=sigma, y=C), fill=NA, color="red", linewidth=1.5,
              inherit.aes=FALSE) +
    scale_fill_gradient2(low="#eff6ff", mid="#93c5fd", high="#1d4ed8",
                         midpoint=mean(res$AUC_CV)) +
    labs(title    = svm_obj$label,
         subtitle = paste0("★ Best: C=", best_row$C,
                           ", σ=", best_row$sigma,
                           " | AUC=", round(best_row$AUC_CV, 4)),
         x="sigma (σ)", y="Cost (C)") +
    theme_bw(base_size=10) +
    theme(plot.title=element_text(face="bold", hjust=0.5),
          plot.subtitle=element_text(size=8, color="gray40", hjust=0.5))
}

gridExtra::grid.arrange(
  grobs = list(plot_svm_grid(svm_none_obj), plot_svm_grid(svm_down3_obj),
               plot_svm_grid(svm_smote_obj), plot_svm_grid(svm_weighted_obj)),
  nrow=2, ncol=2,
  top=grid::textGrob("Tuning — SVM (e1071, RBF Kernel)",
                     gp=grid::gpar(fontsize=13, fontface="bold")))

# ── Plot LASSO lambda ─────────────────────────────────────────────────────────
lasso_res <- lr_lasso$results
ggplot(lasso_res, aes(x=log10(lambda), y=ROC)) +
  geom_ribbon(aes(ymin=ROC-ROCSD, ymax=ROC+ROCSD), alpha=0.15, fill="#2563eb") +
  geom_line(color="#2563eb", linewidth=1) +
  geom_vline(xintercept=log10(lr_lasso$bestTune$lambda), linetype="dashed", color="red") +
  annotate("text", x=log10(lr_lasso$bestTune$lambda)+0.15, y=min(lasso_res$ROC),
           label=paste0("Best λ=", round(lr_lasso$bestTune$lambda,5),
                        "\nAUC=", round(max(lasso_res$ROC),4)),
           color="red", size=3.5, hjust=0) +
  labs(title="LASSO: Lambda Tuning", x="log10(λ)", y="AUC (CV)") +
  theme_bw(base_size=11) + theme(plot.title=element_text(face="bold", hjust=0.5))

# Xuất Excel tuning (chỉ caret models có $results)
write_xlsx(
  lapply(seq_along(caret_model_list), function(i)
    caret_model_list[[i]]$results %>% mutate(Model=caret_model_names[[i]]) %>%
      select(Model, everything())) %>% setNames(caret_model_names),
  "Tuning_Results_RF_LR.xlsx")
write_xlsx(list("XGBoost"=xgb_tuning_summary, "SVM"=svm_tuning_summary),
           "Tuning_Results_XGB_SVM.xlsx")
cat("✓ Xuất Tuning_Results\n")

# ==============================================================================
# PHẦN 12: HÀM ĐÁNH GIÁ
# ==============================================================================
wilson_ci <- function(x, n, conf=0.95) {
  if (is.na(x)||is.na(n)||n==0) return(c(lower=NA_real_, upper=NA_real_))
  z <- qnorm(1-(1-conf)/2); p <- x/n; denom <- 1+z^2/n
  centre <- (p+z^2/(2*n))/denom
  margin <- z*sqrt(p*(1-p)/n+z^2/(4*n^2))/denom
  c(lower=round(max(0,centre-margin),4), upper=round(min(1,centre+margin),4))
}

# FIX: get_prob_safe chỉ dùng cho caret objects (RF/LR)
# XGB dùng get_prob_xgb, SVM dùng get_prob_svm
get_prob_safe <- function(model, x_new) {
  vars_needed <- if (inherits(model$finalModel, "ranger")) {
    model$finalModel$forest$independent.variable.names
  } else if (!is.null(model$finalModel$xNames)) {
    model$finalModel$xNames
  } else {
    names(x_new)
  }
  x_eval  <- x_new
  missing <- setdiff(vars_needed, names(x_eval))
  for (col in missing) x_eval[[col]] <- NA_real_
  prob_raw <- predict(model, newdata=x_eval[, vars_needed, drop=FALSE], type="prob")
  if (is.data.frame(prob_raw)) as.numeric(prob_raw[["Yes"]]) else as.numeric(prob_raw)
}

evaluate_full <- function(prob, y_test, label) {
  roc_o     <- roc(y_test, prob, quiet=TRUE)
  n_pos     <- sum(y_test == "Yes")
  ci_method <- if (n_pos < 50) "bootstrap" else "delong"
  set.seed(123)
  auc_ci  <- ci.auc(roc_o, conf.level=0.95, method=ci_method,
                    boot.n=if(n_pos<50) 2000 else 1000)
  auc_val   <- round(as.numeric(auc_ci[2]),4)
  auc_lower <- round(as.numeric(auc_ci[1]),4)
  auc_upper <- round(as.numeric(auc_ci[3]),4)
  
  thr_youden <- coords(roc_o,"best",ret="threshold",best.method="youden")$threshold[1]
  all_coords <- coords(roc_o,"all",ret=c("threshold","sensitivity","specificity"))
  sens_thr   <- all_coords %>% filter(sensitivity>=0.70) %>%
    arrange(desc(specificity)) %>% slice(1) %>% pull(threshold)
  if (length(sens_thr)==0) sens_thr <- thr_youden
  
  eval_at <- function(thr, thr_name) {
    pred <- factor(ifelse(prob>=thr,"Yes","No"), levels=c("No","Yes"))
    cm   <- confusionMatrix(pred, y_test, positive="Yes")
    tbl  <- cm$table
    TP <- tbl["Yes","Yes"]; FP <- tbl["Yes","No"]
    TN <- tbl["No","No"];   FN <- tbl["No","Yes"]
    
    sens_val <- round(cm$byClass["Sensitivity"],    4)
    spec_val <- round(cm$byClass["Specificity"],    4)
    ppv_val  <- round(cm$byClass["Pos Pred Value"], 4)
    f1_val   <- round(cm$byClass["F1"],             4)
    acc_val  <- round(cm$overall["Accuracy"],       4)
    ci_sens  <- wilson_ci(TP, TP+FN); ci_spec <- wilson_ci(TN, TN+FP)
    ci_ppv   <- wilson_ci(TP, TP+FP); ci_acc  <- wilson_ci(TP+TN, TP+FP+TN+FN)
    
    set.seed(123)
    f1_boot <- replicate(1000, {
      idx    <- sample(length(y_test), replace=TRUE)
      pred_b <- pred[idx]; y_b <- y_test[idx]
      if (length(unique(y_b))<2) return(NA_real_)
      tryCatch(confusionMatrix(pred_b,y_b,positive="Yes")$byClass["F1"],
               error=function(e) NA_real_)
    })
    ci_f1 <- round(quantile(f1_boot, c(0.025,0.975), na.rm=TRUE), 4)
    
    data.frame(
      Model=label, Threshold=thr_name, Thr_Value=round(thr,3),
      AUC=auc_val,
      AUC_display  =sprintf("%.4f (%.4f–%.4f)", auc_val,  auc_lower, auc_upper),
      Sensitivity=sens_val,
      Sens_display =sprintf("%.4f (%.4f–%.4f)", sens_val, ci_sens["lower"], ci_sens["upper"]),
      Specificity=spec_val,
      Spec_display =sprintf("%.4f (%.4f–%.4f)", spec_val, ci_spec["lower"], ci_spec["upper"]),
      PPV=ppv_val,
      PPV_display  =sprintf("%.4f (%.4f–%.4f)", ppv_val,  ci_ppv["lower"],  ci_ppv["upper"]),
      Accuracy=acc_val,
      Acc_display  =sprintf("%.4f (%.4f–%.4f)", acc_val,  ci_acc["lower"],  ci_acc["upper"]),
      F1=f1_val,
      F1_display   =sprintf("%.4f (%.4f–%.4f)", f1_val,   unname(ci_f1[1]), unname(ci_f1[2])),
      row.names=NULL)
  }
  
  bind_rows(eval_at(0.5,"Default (0.5)"), eval_at(thr_youden,"Youden"),
            eval_at(sens_thr,"Sens>=0.70"))
}

# ==============================================================================
# PHẦN 13: ĐÁNH GIÁ TRÊN TEST SET
# FIX: RF/LR → get_prob_safe | XGB → get_prob_xgb | SVM → get_prob_svm
# ==============================================================================
probs_raw <- list(
  "RF None"        = get_prob_safe(rf_none,     x_test_rf),
  "RF Down"        = get_prob_safe(rf_down3,    x_test_rf),
  "RF SMOTE"       = get_prob_safe(rf_smote,    x_test_rf),
  "RF Weighted"    = get_prob_safe(rf_weighted, x_test_rf),
  "XGB None"       = get_prob_xgb(xgb_none_obj,     x_test_xgb),
  "XGB Down"       = get_prob_xgb(xgb_down3_obj,    x_test_xgb),
  "XGB SMOTE"      = get_prob_xgb(xgb_smote_obj,    x_test_xgb),
  "XGB Weighted"   = get_prob_xgb(xgb_weighted_obj, x_test_xgb),
  "SVM None"       = get_prob_svm(svm_none_obj,     x_test_svm),
  "SVM Down"       = get_prob_svm(svm_down3_obj,    x_test_svm),
  "SVM SMOTE"      = get_prob_svm(svm_smote_obj,    x_test_svm),
  "SVM Weighted"   = get_prob_svm(svm_weighted_obj, x_test_svm),
  "Logistic"       = get_prob_safe(lr_model, x_test_lr),
  "Logistic LASSO" = get_prob_safe(lr_lasso, x_test_lr)
)

# Calibrated scores — apply Platt scaler lên raw test scores
probs_list <- list(
  "RF None"        = apply_platt(platt_rf_none,     probs_raw[["RF None"]]),
  "RF Down"        = apply_platt(platt_rf_down3,    probs_raw[["RF Down"]]),
  "RF SMOTE"       = apply_platt(platt_rf_smote,    probs_raw[["RF SMOTE"]]),
  "RF Weighted"    = apply_platt(platt_rf_weighted, probs_raw[["RF Weighted"]]),
  "XGB None"       = apply_platt(platt_xgb_none,     probs_raw[["XGB None"]]),
  "XGB Down"       = apply_platt(platt_xgb_down3,    probs_raw[["XGB Down"]]),
  "XGB SMOTE"      = apply_platt(platt_xgb_smote,    probs_raw[["XGB SMOTE"]]),
  "XGB Weighted"   = apply_platt(platt_xgb_weighted, probs_raw[["XGB Weighted"]]),
  "SVM None"       = apply_platt(platt_svm_none,     probs_raw[["SVM None"]]),
  "SVM Down"       = apply_platt(platt_svm_down3,    probs_raw[["SVM Down"]]),
  "SVM SMOTE"      = apply_platt(platt_svm_smote,    probs_raw[["SVM SMOTE"]]),
  "SVM Weighted"   = apply_platt(platt_svm_weighted, probs_raw[["SVM Weighted"]]),
  "Logistic"       = apply_platt(platt_lr_model, probs_raw[["Logistic"]]),
  "Logistic LASSO" = apply_platt(platt_lr_lasso, probs_raw[["Logistic LASSO"]])
)

# Kiểm tra
cat("=== Kiểm tra calibrated probs_list ===\n")
for (nm in names(probs_list)) {
  p <- probs_list[[nm]]
  cat(sprintf("%-20s | length=%d | NA=%d | range=[%.3f, %.3f]\n",
              nm, length(p), sum(is.na(p)), min(p,na.rm=TRUE), max(p,na.rm=TRUE)))
}

# Kiểm tra trước khi evaluate
cat("=== Kiểm tra probs_list ===\n")
for (nm in names(probs_list)) {
  p <- probs_list[[nm]]
  cat(sprintf("%-20s | length=%d | NA=%d\n", nm, length(p), sum(is.na(p))))
}

cat("\nĐang tính CI cho 14 models...\n")
results_detail <- bind_rows(lapply(names(probs_list), function(nm) {
  cat(" →", nm, "\n"); evaluate_full(probs_list[[nm]], y_test, nm)
}))

summary_auc <- results_detail %>%
  filter(Threshold == "Youden") %>%
  arrange(desc(AUC)) %>%
  mutate(Rank        = row_number(),
         Youden_J    = round(Sensitivity + Specificity - 1, 4),
         Model_Group = case_when(
           startsWith(Model,"RF")  ~ "Random Forest",
           startsWith(Model,"XGB") ~ "XGBoost",
           startsWith(Model,"SVM") ~ "SVM",
           TRUE                    ~ "Logistic"))

write_xlsx(
  list("Full"    = results_detail,
       "Display" = results_detail %>%
         select(Model, Threshold, Thr_Value, AUC_display, Sens_display,
                Spec_display, PPV_display, Acc_display, F1_display),
       "Best_AUC" = summary_auc %>%
         select(Rank, Model, Model_Group, AUC_display, Sens_display,
                Spec_display, PPV_display, F1_display, Youden_J),
       "Best_Per_Group" = summary_auc %>%
         group_by(Model_Group) %>% slice_max(AUC, n=1) %>% ungroup() %>%
         arrange(desc(AUC)) %>%
         select(Model_Group, Model, AUC_display, Sens_display,
                Spec_display, PPV_display, F1_display, Youden_J)),
  "Results_Detail.xlsx")
cat("✓ Xuất Results_Detail.xlsx (4 sheet)\n")

cat("\n=== SO SÁNH 14 MODELS ===\n")
print(summary_auc %>%
        select(Rank, Model, Model_Group, AUC_display, Sens_display, Spec_display, Youden_J))
cat("\n=== BEST MODEL MỖI NHÓM ===\n")
print(summary_auc %>% group_by(Model_Group) %>% slice_max(AUC,n=1) %>% ungroup() %>%
        arrange(desc(AUC)) %>% select(Model_Group, Model, AUC_display, Sens_display, Spec_display))

best_nm_auc <- summary_auc$Model[1]
best_auc    <- summary_auc$AUC[1]
cat(sprintf("\n★ Best model: %s | AUC=%.4f | Sens=%.4f | Spec=%.4f | J=%.4f\n",
            best_nm_auc, best_auc, summary_auc$Sensitivity[1],
            summary_auc$Specificity[1], summary_auc$Youden_J[1]))

# ==============================================================================
# PHẦN 14: VISUALISATIONS
# FIX: 14 màu cho 14 models
# ==============================================================================

# FIX: Tạo đủ 14 màu
colors14 <- c(
  "#1d4ed8","#3b82f6","#93c5fd","#1e40af",   # RF: 4 sắc xanh đậm
  "#16a34a","#4ade80","#bbf7d0","#14532d",   # XGB: 4 sắc xanh lá
  "#dc2626","#f87171","#fecaca","#991b1b",   # SVM: 4 sắc đỏ
  "#d97706","#7c3aed"                        # LR: vàng + tím
)

# 14.1 ROC overlay
roc_list <- lapply(probs_list, function(p) roc(y_test, p, quiet=TRUE))
plot(roc_list[[1]], col=colors14[1], lwd=2, main="ROC — 14 mô hình")
for (i in 2:length(roc_list))
  plot(roc_list[[i]], col=colors14[i], lwd=1.5, add=TRUE)
abline(a=0, b=1, lty=2, col="gray60")
legend("bottomright", bty="n", lwd=2, col=colors14, cex=0.65,
       legend=paste0(names(probs_list)," (", sapply(roc_list, function(r) round(auc(r),4)),")"))

aucs <- sapply(roc_list, function(r) as.numeric(auc(r)))
top5_idx <- order(aucs, decreasing = TRUE)[1:5]
roc_list_top <- roc_list[top5_idx]
names_top    <- names(probs_list)[top5_idx]
colors_top   <- colors14[top5_idx] # Lấy màu tương ứng với mô hình đó
plot(roc_list_top[[1]], col = colors_top[1], lwd = 2.5, 
     main = "Top 5 mô hình có AUC cao nhất")
for (i in 2:length(roc_list_top)) {
  plot(roc_list_top[[i]], col = colors_top[i], lwd = 2, add = TRUE)
}
abline(a = 0, b = 1, lty = 2, col = "gray60")
legend("bottomright", bty = "n", lwd = 2, col = colors_top, cex = 0.8,
       legend = paste0(names_top, " (AUC = ", round(aucs[top5_idx], 4), ")"))

# 14.2 AUC barchart
auc_df <- data.frame(Model=names(probs_list),
                     AUC  =sapply(roc_list, function(r) round(auc(r),4)),
                     Group=case_when(
                       startsWith(names(probs_list),"RF")  ~ "RF",
                       startsWith(names(probs_list),"XGB") ~ "XGB",
                       startsWith(names(probs_list),"SVM") ~ "SVM",
                       TRUE ~ "LR"))

ggplot(auc_df, aes(x=reorder(Model,AUC), y=AUC, fill=Group)) +
  geom_bar(stat="identity", width=0.7) +
  geom_text(aes(label=round(AUC,4)), hjust=-0.1, size=3.2) +
  coord_flip() +
  scale_fill_manual(values=c("RF"="#1d4ed8","XGB"="#16a34a","SVM"="#dc2626","LR"="#d97706")) +
  ylim(0, 1.08) +
  labs(title="So sánh AUC — 14 mô hình", x="", y="AUC (test set)", fill="Nhóm") +
  theme_bw() + theme(plot.title=element_text(face="bold", hjust=0.5))

# 14.3 Sensitivity vs Specificity
ggplot(summary_auc, aes(x=Specificity, y=Sensitivity, color=Model_Group, shape=Model_Group)) +
  geom_point(size=3.5, alpha=0.9) +
  ggrepel::geom_label_repel(aes(label=Model), size=2.5, max.overlaps=Inf) +
  scale_color_manual(values=c("Random Forest"="#1d4ed8","XGBoost"="#16a34a",
                              "SVM"="#dc2626","Logistic"="#d97706")) +
  geom_vline(xintercept=0.6, linetype="dashed", alpha=0.5) +
  geom_hline(yintercept=0.7, linetype="dashed", alpha=0.5) +
  annotate("rect", xmin=0.6, xmax=1, ymin=0.7, ymax=1, fill="green", alpha=0.05) +
  annotate("text", x=0.8, y=0.72, label="Vùng mục tiêu\nSens≥0.7 & Spec≥0.6",
           color="darkgreen", size=3.2) +
  labs(title="Sensitivity vs Specificity (Youden threshold)",
       color="Nhóm model", shape="Nhóm model") +
  theme_bw() + theme(plot.title=element_text(face="bold", hjust=0.5))

# 14.4 Variable Importance — chỉ RF và LR có varImp() trong caret
# FIX: Kiểm tra best model có phải caret object không
# FIX: Tạo named list đầy đủ — RF/LR dùng caret object, XGB/SVM dùng NULL
caret_named <- setNames(caret_model_list, caret_model_names)

if (best_nm_auc %in% names(caret_named)) {
  # Best model là RF hoặc LR → varImp() được
  plot(varImp(caret_named[[best_nm_auc]], scale=TRUE), top=15,
       main=paste0("Variable Importance — ", best_nm_auc))
} else {
  # Best model là XGB hoặc SVM → dùng XGB importance hoặc skip
  cat("⚠ Best model là", best_nm_auc, "— không có varImp() từ caret\n")
  cat("  Dùng SHAP (Phần 16) để phân tích feature importance\n")
  
  # Nếu best là XGB → vẽ XGBoost native importance
  if (startsWith(best_nm_auc, "XGB")) {
    xgb_obj_best <- list(xgb_none_obj, xgb_down3_obj,
                         xgb_smote_obj, xgb_weighted_obj) %>%
      setNames(c("XGB None","XGB Down","XGB SMOTE","XGB Weighted"))
    imp_mat <- xgb.importance(model=xgb_obj_best[[best_nm_auc]]$model)
    xgb.plot.importance(imp_mat, top_n=15,
                        main=paste0("XGBoost Feature Importance — ", best_nm_auc))
  }
}

# 14.5 CV Resamples — chỉ RF + LR (cùng fold idx_none)
resamps <- resamples(list(RF_None=rf_none, RF_Weighted=rf_weighted,
                          Logistic=lr_model, Logistic_LASSO=lr_lasso))
bwplot(resamps, metric="ROC", main="CV AUC Distribution (RF + LR)")

# ==============================================================================
# PHẦN 15: CHẨN ĐOÁN CHẤT LƯỢNG DATA
# ==============================================================================
cat("\n=== UNIVARIATE ANALYSIS ===\n")
univariate_results <- lapply(names(x_train_fs), function(var) {
  x_var    <- as.numeric(x_train_fs[[var]])
  y_var    <- ifelse(y_train=="Yes",1,0)
  n_unique <- length(unique(na.omit(x_var)))
  p_val    <- tryCatch(
    if (n_unique<=2) fisher.test(table(x_var,y_var))$p.value
    else wilcox.test(x_var~y_var)$p.value,
    error=function(e) NA_real_)
  auc_val  <- tryCatch(as.numeric(auc(roc(y_train,x_var,quiet=TRUE))),
                       error=function(e) 0.5)
  data.frame(Variable=var, p_value=round(p_val,4), AUC_univariate=round(auc_val,3))
}) %>% bind_rows() %>% arrange(p_value)

cat("Top 15:\n"); print(head(univariate_results,15))
leakage <- univariate_results %>% filter(AUC_univariate>0.85)
if (nrow(leakage)>0) { cat("⚠ Nghi ngờ leakage:\n"); print(leakage) } else
  cat("✓ Không phát hiện leakage\n")

cat("\n=== UPPER BOUND ===\n")
x_all <- bind_rows(x_train_fs, x_test_fs); y_all <- c(y_train, y_test)
set.seed(123)
rf_ub  <- train(x=x_all, y=y_all, method="ranger",
                trControl=trainControl(method="none", classProbs=TRUE),
                tuneGrid=rf_down3$bestTune, num.trees=500)
auc_ub <- round(auc(roc(y_all, predict(rf_ub,x_all,type="prob")$Yes, quiet=TRUE)),4)
cat("AUC upper bound:", auc_ub, if(auc_ub<0.70) "⚠ Data yếu" else "✓ Data có tín hiệu", "\n")

nzv      <- nearZeroVar(x_train_fs, saveMetrics=TRUE)
high_cor <- findCorrelation(cor(as.matrix(x_train_fs), use="complete.obs"), cutoff=0.85)
cat(sprintf("NZV: %d | Cor>0.85: %d | NA cols: %d\n",
            sum(nzv$nzv), length(high_cor), sum(colSums(is.na(x_train_fs))>0)))

# ==============================================================================
# PHẦN 16: SHAP ANALYSIS — kernelshap + shapviz
# FIX: Handle cả RF, XGB (manual), SVM (e1071) — mỗi loại có pred_fun riêng
# FIX: Xoá key trùng trong labels_mapping
# ==============================================================================
# ==============================================================================
# PHẦN 16: SHAP ANALYSIS — fastshap
# Dùng fastshap::explain() — nhẹ hơn kernelshap, không cần bg_X
# Hỗ trợ RF (caret), XGB (manual), SVM (e1071) qua pred_fun wrapper
# ==============================================================================
library(fastshap)
library(shapviz)

cat("\n=== SHAP — Best model:", best_nm_auc, "===\n")

# Xác định x_shap_train, x_shap_test, pred_fun theo loại best model
if (best_nm_auc %in% caret_model_names) {
  
  # ── RF hoặc LR (caret) ──────────────────────────────────────────────────────
  best_model_obj <- caret_named[[best_nm_auc]]
  
  is_rf <- inherits(best_model_obj$finalModel, "ranger")
  is_lr <- best_nm_auc %in% c("Logistic", "Logistic LASSO")
  
  if (is_rf) {
    vars_shap  <- best_model_obj$finalModel$forest$independent.variable.names
    x_shap_tr  <- as.data.frame(x_train_rf[, vars_shap, drop=FALSE])
    x_shap_te  <- as.data.frame(x_test_rf[,  vars_shap, drop=FALSE])
  } else {
    vars_shap  <- best_model_obj$finalModel$xNames
    vars_shap  <- intersect(vars_shap, names(x_train_lr))
    x_shap_tr  <- as.data.frame(x_train_lr[, vars_shap, drop=FALSE])
    x_shap_te  <- as.data.frame(x_test_lr[,  vars_shap, drop=FALSE])
  }
  
  pred_fun_shap <- function(object, newdata) {
    newdata <- as.data.frame(newdata)
    as.numeric(predict(object, newdata=newdata, type="prob")[["Yes"]])
  }
  shap_object <- best_model_obj
  
} else if (startsWith(best_nm_auc, "XGB")) {
  
  # ── XGBoost manual ──────────────────────────────────────────────────────────
  xgb_objs <- setNames(
    list(xgb_none_obj, xgb_down3_obj, xgb_smote_obj, xgb_weighted_obj),
    c("XGB None","XGB Down","XGB SMOTE","XGB Weighted"))
  best_xgb   <- xgb_objs[[best_nm_auc]]
  vars_shap  <- best_xgb$col_names
  x_shap_tr  <- as.data.frame(x_train_xgb[, vars_shap, drop=FALSE])
  x_shap_te  <- as.data.frame(x_test_xgb[,  vars_shap, drop=FALSE])
  shap_object <- best_xgb$model
  
  pred_fun_shap <- function(object, newdata) {
    as.numeric(predict(object,
                       xgb.DMatrix(as.matrix(newdata))))
  }
  
} else {
  
  # ── SVM e1071 manual ─────────────────────────────────────────────────────────
  svm_objs <- setNames(
    list(svm_none_obj, svm_down3_obj, svm_smote_obj, svm_weighted_obj),
    c("SVM None","SVM Down","SVM SMOTE","SVM Weighted"))
  best_svm   <- svm_objs[[best_nm_auc]]
  vars_shap  <- best_svm$col_names
  x_shap_tr  <- as.data.frame(x_train_svm[, vars_shap, drop=FALSE])
  x_shap_te  <- as.data.frame(x_test_svm[,  vars_shap, drop=FALSE])
  shap_object <- best_svm$model
  
  pred_fun_shap <- function(object, newdata) {
    prob_mat <- attr(
      predict(object, newdata=as.matrix(newdata), probability=TRUE),
      "probabilities")
    as.numeric(prob_mat[, "Yes"])
  }
}

# Subsample để tính nhanh hơn
set.seed(123)
n_bg   <- min(100, nrow(x_shap_tr))
n_eval <- min(100, nrow(x_shap_te))
x_bg   <- x_shap_tr[sample(nrow(x_shap_tr), n_bg), , drop=FALSE]
x_eval <- x_shap_te[sample(nrow(x_shap_te), n_eval), , drop=FALSE]

cat(sprintf("Background: %d | Eval: %d | Features: %d\n",
            nrow(x_bg), nrow(x_eval), length(vars_shap)))

# Tính SHAP bằng fastshap::explain()
# nsim: số Monte Carlo simulations — tăng để chính xác hơn nhưng chậm hơn
set.seed(123)
shap_values <- fastshap::explain(
  object       = shap_object,
  X            = x_bg,           # Background dataset
  pred_wrapper = pred_fun_shap,
  nsim         = 50,
  newdata      = x_eval,
  adjust       = TRUE)           # Điều chỉnh để tổng SHAP = prediction - baseline

# Chuyển sang shapviz để plot
sv <- shapviz(shap_values, X = x_eval)
cat("✓ SHAP values tính xong\n")

# SHAP Bar Plot
p_bar <- sv_importance(sv, kind="bar", max_display=15, fill="#2563eb")

# SHAP Beeswarm
p_bee <- sv_importance(sv, kind="beeswarm", max_display=15, alpha=0.6, size=1.5)

# Mapping nhãn tiếng Việt
labels_mapping <- c(
  "Phác_đồ.Paclitaxel.carboplatin" = "Regimen (Paclitaxel-carboplatin)",
  "Nguy.cơ.phác.đồ"               = "Regimen risk",
  "Phác_đồ.TC.21"                  = "Regimen (TC-21)",
  "Loại_ung_thư.Vú"                = "Cancer type (Vú)",
  "Quy.đổi.dự.phòng"              = "Characteristics of prophylaxis time",
  "Phác_đồ.MTX"                   = "Regimen (MTX)",
  "YTNC.phẫu.thuật"               = "Risk - Surgu",
  "Loại.GCSF.dự.phòng"            = "Loại G-CSF dự phòng",
  "Chu.kỳ"                        = "Chu kỳ",
  "Loại_ung_thư.U.Lympho"         = "Loại ung thư (U Lympho)",
  "Thời.gian.dự.phòng"            = "Thời gian dự phòng",
  "Giới"                          = "Giới tính",
  "Tuổi"                          = "Tuổi",
  "Tiền.sử.giảm.BCTT"             = "Tiền sử giảm BCTT",
  "Số.lượng.YTNC"                 = "Số lượng YTNC",
  "Phác_đồ.AC.14"                 = "Phác đồ (AC-14)",
  "Phác_đồ.AC.21"                 = "Phác đồ (AC-21)"
)

p_bar_updated <- p_bar +
  scale_y_discrete(labels=labels_mapping) +
  labs(x="mean(|SHAP value|)", y="Biến số")

p_bee_updated <- p_bee +
  scale_y_discrete(labels=labels_mapping) +
  labs(x="SHAP value", y=NULL) +
  scale_color_gradient(low="#2563eb", high="#dc2626",
                       breaks=c(0,1), labels=c("Low","High"),
                       name="Feature value") +
  theme(legend.position="right")

gridExtra::grid.arrange(
  p_bar_updated, p_bee_updated, nrow=1, ncol=2,
  top=grid::textGrob(
    paste0("SHAP Analysis — ", best_nm_auc, " | AUC=", round(best_auc,4)),
    gp=grid::gpar(fontsize=13, fontface="bold")))

# SHAP Dependence — top 3 features
top3 <- names(sort(colMeans(abs(sv$S)), decreasing=TRUE))[1:min(3, ncol(sv$S))]
cat("Top 3 SHAP features:", paste(top3, collapse=", "), "\n")

dep_plots <- lapply(top3, function(feat) {
  lbl <- if (feat %in% names(labels_mapping)) labels_mapping[[feat]] else feat
  sv_dependence(sv, v=feat, alpha=0.5) +
    labs(title=lbl, x=lbl, y="SHAP value") +
    theme_bw(base_size=10) +
    theme(plot.title=element_text(face="bold", hjust=0.5))
})
gridExtra::grid.arrange(
  grobs=dep_plots, nrow=1, ncol=length(dep_plots),
  top=grid::textGrob("SHAP Dependence — Top 3 Features",
                     gp=grid::gpar(fontsize=13, fontface="bold")))

# ==============================================================================
# PHẦN 17: CALIBRATION PLOT
# Đánh giá độ hiệu chỉnh xác suất — xác suất dự đoán có khớp thực tế không
#
# Phương pháp: Platt's binning calibration (10 bins)
# Đường calibration tốt: nằm gần đường diagonal y = x
# Brier Score: sai số bình phương trung bình — nhỏ hơn = tốt hơn (0 = hoàn hảo)
# ==============================================================================
library(ggplot2); library(dplyr); library(gridExtra)

# Hàm tính calibration data cho 1 model
calc_calibration <- function(prob, y_obs, n_bins = 10, label = "") {
  y_num  <- as.integer(y_obs == "Yes")
  breaks <- quantile(prob, probs = seq(0, 1, length.out = n_bins + 1), na.rm = TRUE)
  breaks <- unique(breaks)                        # Loại duplicate nếu có
  bin    <- cut(prob, breaks = breaks, include.lowest = TRUE, labels = FALSE)
  
  cal_df <- data.frame(prob = prob, y = y_num, bin = bin) %>%
    group_by(bin) %>%
    summarise(
      mean_pred = mean(prob, na.rm = TRUE),   # Xác suất dự đoán trung bình của bin
      mean_obs  = mean(y,    na.rm = TRUE),   # Tỷ lệ thực sự bị biến cố trong bin
      n         = n(),
      .groups   = "drop") %>%
    mutate(Model = label)
  
  # Brier Score: mean((p - y)^2)
  brier <- mean((prob - y_num)^2, na.rm = TRUE)
  
  list(cal_df = cal_df, brier = round(brier, 4))
}

# Tính calibration cho tất cả 14 models
cal_results <- lapply(names(probs_list), function(nm) {
  res <- calc_calibration(probs_list[[nm]], y_test, n_bins = 10, label = nm)
  res$cal_df$Brier <- res$brier
  res
})
names(cal_results) <- names(probs_list)

# Bảng Brier Score tổng hợp
brier_df <- data.frame(
  Model       = names(cal_results),
  Brier_Score = sapply(cal_results, function(x) x$brier),
  Model_Group = case_when(
    startsWith(names(cal_results), "RF")  ~ "Random Forest",
    startsWith(names(cal_results), "XGB") ~ "XGBoost",
    startsWith(names(cal_results), "SVM") ~ "SVM",
    TRUE                                  ~ "Logistic")
) %>% arrange(Brier_Score)

cat("\n=== BRIER SCORE (nhỏ hơn = calibration tốt hơn) ===\n")
print(brier_df)

# Baseline Brier Score: model luôn dự đoán prevalence
prev      <- mean(as.integer(y_test == "Yes"))
brier_ref <- round(prev * (1 - prev), 4)
cat(sprintf("Baseline Brier (predict prevalence=%.3f): %.4f\n", prev, brier_ref))

# Plot calibration — 4 panels theo nhóm model
cal_all <- bind_rows(lapply(cal_results, function(x) x$cal_df)) %>%
  mutate(Model_Group = case_when(
    startsWith(Model, "RF")  ~ "Random Forest",
    startsWith(Model, "XGB") ~ "XGBoost",
    startsWith(Model, "SVM") ~ "SVM",
    TRUE                     ~ "Logistic"))

group_colors <- c(
  "RF None"="olean#1d4ed8", "RF Down"="#3b82f6",
  "RF SMOTE"="#93c5fd",  "RF Weighted"="#1e40af",
  "XGB None"="#16a34a",  "XGB Down"="#4ade80",
  "XGB SMOTE"="#bbf7d0", "XGB Weighted"="#14532d",
  "SVM None"="#dc2626",  "SVM Down"="#f87171",
  "SVM SMOTE"="#fecaca", "SVM Weighted"="#991b1b",
  "Logistic"="#d97706",  "Logistic LASSO"="#7c3aed")

# Sửa lại màu (bỏ typo "olean#")
group_colors <- c(
  "RF None"="#1d4ed8",    "RF Down"="#3b82f6",
  "RF SMOTE"="#93c5fd",   "RF Weighted"="#1e40af",
  "XGB None"="#16a34a",   "XGB Down"="#4ade80",
  "XGB SMOTE"="#bbf7d0",  "XGB Weighted"="#14532d",
  "SVM None"="#dc2626",   "SVM Down"="#f87171",
  "SVM SMOTE"="#fecaca",  "SVM Weighted"="#991b1b",
  "Logistic"="#d97706",   "Logistic LASSO"="#7c3aed")

plot_cal_group <- function(data, group_title) {
  ggplot(data, aes(x = mean_pred, y = mean_obs, color = Model)) +
    geom_abline(slope = 1, intercept = 0,
                linetype = "dashed", color = "gray50", linewidth = 0.8) +   # Đường lý tưởng
    geom_line(linewidth = 0.8, alpha = 0.8) +
    geom_point(aes(size = n), alpha = 0.8) +
    scale_size_continuous(range = c(1.5, 4), guide = "none") +
    scale_color_manual(values = group_colors) +
    scale_x_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.2)) +
    scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.2)) +
    labs(title    = group_title,
         subtitle = paste0("Brier: ",
                           paste(sapply(unique(data$Model), function(m) {
                             b <- brier_df$Brier_Score[brier_df$Model == m]
                             paste0(m, "=", b)
                           }), collapse = " | ")),
         x = "Xác suất dự đoán trung bình",
         y = "Tỷ lệ thực tế",
         color = NULL) +
    theme_bw(base_size = 10) +
    theme(plot.title    = element_text(face = "bold", hjust = 0.5),
          plot.subtitle = element_text(size = 6.5, color = "gray40", hjust = 0.5),
          legend.position = "bottom",
          legend.text = element_text(size = 8))
}

gridExtra::grid.arrange(
  plot_cal_group(cal_all %>% filter(Model_Group == "Random Forest"), "Calibration — RF"),
  plot_cal_group(cal_all %>% filter(Model_Group == "XGBoost"),       "Calibration — XGBoost"),
  plot_cal_group(cal_all %>% filter(Model_Group == "SVM"),           "Calibration — SVM"),
  plot_cal_group(cal_all %>% filter(Model_Group == "Logistic"),      "Calibration — Logistic"),
  nrow = 2, ncol = 2,
  top  = grid::textGrob(
    "Calibration Plot — 14 Mô Hình (điểm lớn hơn = bin nhiều quan sát hơn)",
    gp = grid::gpar(fontsize = 13, fontface = "bold")))

# So sánh Brier Score trước và sau calibration
cal_results_raw <- lapply(names(probs_raw), function(nm) {
  res <- calc_calibration(probs_raw[[nm]], y_test, n_bins=10, label=nm)
  res$brier
})
names(cal_results_raw) <- names(probs_raw)

brier_compare <- data.frame(
  Model        = names(probs_raw),
  Brier_Raw    = unlist(cal_results_raw),
  Brier_Platt  = sapply(cal_results, function(x) x$brier)
) %>%
  mutate(Delta = round(Brier_Raw - Brier_Platt, 4),
         Improved = ifelse(Delta > 0, "✓ Tốt hơn", "✗ Không cải thiện")) %>%
  arrange(Brier_Platt)

cat("\n=== So sánh Brier Score: Raw vs Platt Calibrated ===\n")
print(brier_compare)

# ==============================================================================
# PHẦN 17.1: CALIBRATION PLOT — Best model mỗi nhóm
# ==============================================================================
library(ggplot2); library(dplyr); library(gridExtra)

# Hàm tính calibration data
calc_calibration <- function(prob, y_obs, n_bins = 10, label = "") {
  y_num  <- as.integer(y_obs == "Yes")
  breaks <- quantile(prob, probs = seq(0, 1, length.out = n_bins + 1), na.rm = TRUE)
  breaks <- unique(breaks)
  bin    <- cut(prob, breaks = breaks, include.lowest = TRUE, labels = FALSE)
  
  cal_df <- data.frame(prob = prob, y = y_num, bin = bin) %>%
    group_by(bin) %>%
    summarise(
      mean_pred = mean(prob, na.rm = TRUE),
      mean_obs  = mean(y,    na.rm = TRUE),
      n         = n(),
      .groups   = "drop") %>%
    mutate(Model = label)
  
  brier <- mean((prob - y_num)^2, na.rm = TRUE)
  list(cal_df = cal_df, brier = round(brier, 4))
}

# Lấy best model mỗi nhóm từ summary_auc (đã có sẵn từ Phần 13)
best_per_group <- summary_auc %>%
  group_by(Model_Group) %>%
  slice_max(AUC, n = 1) %>%
  ungroup() %>%
  arrange(desc(AUC)) %>%
  select(Model, Model_Group, AUC)

cat("\n=== CALIBRATION — Best model mỗi nhóm ===\n")
print(best_per_group)

# Màu cho 4 nhóm
group_color_map <- c(
  "Random Forest" = "#1d4ed8",
  "XGBoost"       = "#16a34a",
  "SVM"           = "#dc2626",
  "Logistic"      = "#d97706")

# Tính calibration cho best model mỗi nhóm
# Raw (chưa calibrate)
cal_raw_best <- lapply(seq_len(nrow(best_per_group)), function(i) {
  nm  <- best_per_group$Model[i]
  res <- calc_calibration(probs_raw[[nm]], y_test, n_bins = 10, label = nm)
  res$cal_df$Type        <- "Trước Platt"
  res$cal_df$Model_Group <- best_per_group$Model_Group[i]
  res$cal_df$Brier       <- res$brier
  res$cal_df
}) %>% bind_rows()

# Calibrated (sau Platt scaling)
cal_platt_best <- lapply(seq_len(nrow(best_per_group)), function(i) {
  nm  <- best_per_group$Model[i]
  res <- calc_calibration(probs_list[[nm]], y_test, n_bins = 10, label = nm)
  res$cal_df$Type        <- "Sau Platt"
  res$cal_df$Model_Group <- best_per_group$Model_Group[i]
  res$cal_df$Brier       <- res$brier
  res$cal_df
}) %>% bind_rows()

cal_best_all <- bind_rows(cal_raw_best, cal_platt_best)

# Bảng Brier Score tổng hợp
brier_best <- data.frame(
  Model       = best_per_group$Model,
  Model_Group = best_per_group$Model_Group,
  AUC         = best_per_group$AUC,
  Brier_Raw   = sapply(best_per_group$Model, function(nm)
    calc_calibration(probs_raw[[nm]],  y_test, label=nm)$brier),
  Brier_Platt = sapply(best_per_group$Model, function(nm)
    calc_calibration(probs_list[[nm]], y_test, label=nm)$brier)
) %>%
  mutate(Delta    = round(Brier_Raw - Brier_Platt, 4),
         Improved = ifelse(Delta > 0, "✓ Tốt hơn", "✗ Không đổi")) %>%
  arrange(Brier_Platt)

cat("\n=== Brier Score: Raw vs Platt ===\n")
print(brier_best)

prev      <- mean(as.integer(y_test == "Yes"))
brier_ref <- round(prev * (1 - prev), 4)
cat(sprintf("Baseline Brier (prevalence=%.3f): %.4f\n", prev, brier_ref))

# Plot 1: 4 panels — mỗi panel 1 nhóm, 2 đường (trước/sau Platt)
plot_cal_one <- function(group_name) {
  data  <- cal_best_all %>% filter(Model_Group == group_name)
  nm    <- unique(data$Model)
  color <- group_color_map[group_name]
  
  b_raw   <- brier_best$Brier_Raw[brier_best$Model_Group   == group_name]
  b_platt <- brier_best$Brier_Platt[brier_best$Model_Group == group_name]
  
  ggplot(data, aes(x = mean_pred, y = mean_obs,
                   color = Type, linetype = Type)) +
    geom_abline(slope = 1, intercept = 0,
                linetype = "dotted", color = "gray50", linewidth = 0.7) +
    geom_line(linewidth = 0.9, alpha = 0.9) +
    geom_point(aes(size = n), alpha = 0.85) +
    scale_size_continuous(range = c(1.5, 4.5), guide = "none") +
    scale_color_manual(values = c("Trước Platt" = "gray60",
                                  "Sau Platt"   = color)) +
    scale_linetype_manual(values = c("Trước Platt" = "dashed",
                                     "Sau Platt"   = "solid")) +
    scale_x_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.2)) +
    scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.2)) +
    labs(
      title    = paste0(group_name, " — ", nm),
      subtitle = sprintf(
        "Brier: %.4f → %.4f (Δ=%.4f) | AUC=%.4f",
        b_raw, b_platt, b_raw - b_platt,
        brier_best$AUC[brier_best$Model_Group == group_name]),
      x     = "Xác suất dự đoán",
      y     = "Tỷ lệ thực tế",
      color = NULL, linetype = NULL) +
    theme_bw(base_size = 10) +
    theme(
      plot.title    = element_text(face = "bold", hjust = 0.5, size = 11),
      plot.subtitle = element_text(size = 7.5, color = "gray40", hjust = 0.5),
      legend.position = "bottom",
      legend.text     = element_text(size = 9))
}

gridExtra::grid.arrange(
  plot_cal_one("Random Forest"),
  plot_cal_one("XGBoost"),
  plot_cal_one("SVM"),
  plot_cal_one("Logistic"),
  nrow = 2, ncol = 2,
  top  = grid::textGrob(
    "Calibration Plot — Best Model Mỗi Nhóm (Trước vs Sau Platt Scaling)",
    gp = grid::gpar(fontsize = 13, fontface = "bold")))

# Plot 2: Overlay 4 best models sau Platt trên cùng 1 panel
cal_platt_overlay <- cal_platt_best %>%
  left_join(brier_best %>% select(Model, Brier_Platt, AUC),
            by = "Model") %>%
  mutate(Label = paste0(Model,
                        "\nBrier=", round(Brier_Platt, 4)))

ggplot(cal_platt_overlay,
       aes(x = mean_pred, y = mean_obs,
           color = Model_Group, group = Model)) +
  geom_abline(slope = 1, intercept = 0,
              linetype = "dotted", color = "gray50", linewidth = 0.8) +
  geom_line(linewidth = 1, alpha = 0.9) +
  geom_point(aes(size = n), alpha = 0.85) +
  scale_size_continuous(range = c(1.5, 4.5), guide = "none") +
  scale_color_manual(
    values = group_color_map,
    labels = sapply(names(group_color_map), function(g) {
      row <- brier_best[brier_best$Model_Group == g, ]
      if (nrow(row) == 0) return(g)
      paste0(row$Model, "\n(Brier=", row$Brier_Platt,
             " | AUC=", round(row$AUC, 3), ")")
    })) +
  scale_x_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.2)) +
  scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.2)) +
  labs(
    title    = "Calibration — Best Model Mỗi Nhóm (Sau Platt Scaling)",
    subtitle = paste0("Đường chấm = calibration lý tưởng | ",
                      "Baseline Brier = ", brier_ref,
                      " (predict prevalence=", round(prev*100,1), "%)"),
    x     = "Xác suất dự đoán",
    y     = "Tỷ lệ thực tế",
    color = "Nhóm model") +
  theme_bw(base_size = 11) +
  theme(
    plot.title      = element_text(face = "bold", hjust = 0.5, size = 12),
    plot.subtitle   = element_text(size = 8, color = "gray40", hjust = 0.5),
    legend.position = "right",
    legend.text     = element_text(size = 8),
    legend.key.height = unit(1.2, "cm"))

# Xuất Excel
write_xlsx(
  list("Brier_Summary" = brier_best,
       "Cal_Raw"       = cal_raw_best   %>% select(-Brier),
       "Cal_Platt"     = cal_platt_best %>% select(-Brier)),
  "Calibration_Best.xlsx")
cat("✓ Xuất Calibration_Best.xlsx\n")

# ==============================================================================
# PHẦN 18: DECISION CURVE ANALYSIS (DCA)
# Đánh giá lợi ích lâm sàng thực tế của model theo từng ngưỡng quyết định
#
# Net Benefit = TP/n − FP/n × (pt/(1−pt))
#   pt = probability threshold (ngưỡng bác sĩ chấp nhận can thiệp)
#   n  = tổng số bệnh nhân
#
# Đường "Treat All": can thiệp tất cả mọi người
# Đường "Treat None": không can thiệp ai → Net Benefit = 0
# Model tốt: đường NB nằm TRÊN cả "Treat All" và "Treat None"
#
# Tài liệu: Vickers AJ, Elkin EB. Decision curve analysis: a novel method for
#           evaluating prediction models. Med Decis Making. 2006;26(6):565-574.
# ==============================================================================

# Hàm tính Net Benefit tại 1 ngưỡng pt
net_benefit_at <- function(prob, y_obs, pt) {
  y_num <- as.integer(y_obs == "Yes")
  n     <- length(y_num)
  pred  <- as.integer(prob >= pt)
  TP    <- sum(pred == 1 & y_num == 1)
  FP    <- sum(pred == 1 & y_num == 0)
  TP/n - FP/n * (pt / (1 - pt))
}

# Tính DCA trên dải ngưỡng [0.01, 0.50]
# Giới hạn 0.50 vì ngưỡng > prevalence thường không có ý nghĩa lâm sàng
pt_seq <- seq(0.01, 0.50, by = 0.01)

# Net benefit cho từng model
dca_model <- lapply(names(probs_list), function(nm) {
  nb <- sapply(pt_seq, function(pt)
    net_benefit_at(probs_list[[nm]], y_test, pt))
  data.frame(pt = pt_seq, NB = nb, Model = nm,
             Model_Group = case_when(
               startsWith(nm, "RF")  ~ "Random Forest",
               startsWith(nm, "XGB") ~ "XGBoost",
               startsWith(nm, "SVM") ~ "SVM",
               TRUE                  ~ "Logistic"))
}) %>% bind_rows()

# Net benefit cho "Treat All" và "Treat None"
y_num  <- as.integer(y_test == "Yes")
n_test <- length(y_num)
prev   <- mean(y_num)

dca_ref <- data.frame(
  pt    = pt_seq,
  # Treat All: TP = tất cả bệnh thật, FP = tất cả người khoẻ
  NB_treat_all  = prev - (1 - prev) * (pt_seq / (1 - pt_seq)),
  NB_treat_none = 0)    # Treat None luôn = 0

# Plot DCA — overlay tất cả models + 2 đường reference
# Dùng nhóm màu nhất quán với toàn pipeline
plot_dca_group <- function(group_name, group_label) {
  data_group <- dca_model %>% filter(Model_Group == group_name)
  
  ggplot() +
    # Treat None (luôn = 0)
    geom_hline(yintercept = 0, linetype = "dotted",
               color = "black", linewidth = 0.8) +
    # Treat All
    geom_line(data = dca_ref, aes(x = pt, y = NB_treat_all),
              linetype = "longdash", color = "gray40", linewidth = 0.9) +
    # Các model
    geom_line(data = data_group,
              aes(x = pt, y = NB, color = Model), linewidth = 0.9, alpha = 0.9) +
    scale_color_manual(values = group_colors) +
    # Vùng prevalence thực tế (ngưỡng hợp lý nhất)
    geom_vline(xintercept = prev, linetype = "dashed",
               color = "#dc2626", alpha = 0.6, linewidth = 0.7) +
    annotate("text", x = prev + 0.01, y = max(dca_ref$NB_treat_all, na.rm = TRUE),
             label = paste0("Prevalence\n", round(prev, 3)),
             color = "#dc2626", size = 2.8, hjust = 0) +
    scale_x_continuous(limits = c(0, 0.50),
                       breaks = seq(0, 0.50, 0.10),
                       labels = scales::percent_format(accuracy = 1)) +
    coord_cartesian(ylim = c(-0.02, max(dca_ref$NB_treat_all, na.rm = TRUE) * 1.1)) +
    labs(title    = group_label,
         subtitle = "--- Treat All  ··· Treat None  | Đường đỏ = Prevalence",
         x        = "Ngưỡng xác suất (Threshold)",
         y        = "Net Benefit",
         color    = NULL) +
    theme_bw(base_size = 10) +
    theme(plot.title    = element_text(face = "bold", hjust = 0.5),
          plot.subtitle = element_text(size = 7, color = "gray40", hjust = 0.5),
          legend.position = "bottom",
          legend.text     = element_text(size = 8))
}

gridExtra::grid.arrange(
  plot_dca_group("Random Forest", "DCA — Random Forest"),
  plot_dca_group("XGBoost",       "DCA — XGBoost"),
  plot_dca_group("SVM",           "DCA — SVM"),
  plot_dca_group("Logistic",      "DCA — Logistic"),
  nrow = 2, ncol = 2,
  top  = grid::textGrob(
    "Decision Curve Analysis — 14 Mô Hình",
    gp = grid::gpar(fontsize = 13, fontface = "bold")))

# Plot DCA tổng hợp — chỉ best model mỗi nhóm + 2 reference
best_per_group <- summary_auc %>%
  group_by(Model_Group) %>%
  slice_max(AUC, n = 1) %>%
  ungroup() %>%
  pull(Model)

dca_best <- dca_model %>% filter(Model %in% best_per_group)

ggplot() +
  geom_hline(yintercept = 0, linetype = "dotted",
             color = "black", linewidth = 0.8) +
  annotate("text", x = 0.48, y = 0.002,
           label = "Treat None", color = "black", size = 3, hjust = 1) +
  geom_line(data = dca_ref, aes(x = pt, y = NB_treat_all),
            linetype = "longdash", color = "gray40", linewidth = 1) +
  annotate("text", x = 0.48,
           y = net_benefit_at(rep(1, n_test), y_test, 0.48) + 0.005,
           label = "Treat All", color = "gray40", size = 3, hjust = 1) +
  geom_line(data = dca_best,
            aes(x = pt, y = NB, color = Model), linewidth = 1.2, alpha = 0.9) +
  geom_vline(xintercept = prev, linetype = "dashed",
             color = "#dc2626", alpha = 0.7, linewidth = 0.8) +
  annotate("text", x = prev + 0.012,
           y = max(dca_ref$NB_treat_all, na.rm = TRUE) * 0.95,
           label = paste0("Prevalence = ", round(prev * 100, 1), "%"),
           color = "#dc2626", size = 3, hjust = 0) +
  scale_x_continuous(limits = c(0, 0.50), breaks = seq(0, 0.50, 0.10),
                     labels = scales::percent_format(accuracy = 1)) +
  coord_cartesian(ylim = c(-0.02, max(dca_ref$NB_treat_all, na.rm = TRUE) * 1.15)) +
  labs(title    = "Decision Curve Analysis — Best Model Mỗi Nhóm",
       subtitle = paste0("Net Benefit > Treat All & Treat None → model mang lại lợi ích lâm sàng\n",
                         "Vùng ngưỡng lâm sàng hợp lý: quanh prevalence (đường đỏ)"),
       x        = "Threshold",
       y        = "Net benefit",
       color    = "Model (best/group)") +
  theme_bw(base_size = 11) +
  theme(plot.title    = element_text(face = "bold", hjust = 0.5, size = 13),
        plot.subtitle = element_text(size = 8, color = "gray40", hjust = 0.5),
        legend.position = "bottom",
        legend.text     = element_text(size = 9))

# Xuất bảng DCA tóm tắt tại ngưỡng prevalence và 0.20
dca_summary <- dca_model %>%
  filter(pt %in% c(round(prev, 2), 0.10, 0.20, 0.30)) %>%
  select(Model, pt, NB, Model_Group) %>%
  pivot_wider(names_from = pt, values_from = NB, 
              names_prefix = "NB_pt") %>%
  left_join(brier_df %>% select(Model, Brier_Score), by = "Model") %>%
  arrange(Model_Group, desc(across(starts_with("NB_pt"), ~ .x, .names = "{.col}")))

write_xlsx(
  list("Calibration_Brier" = brier_df,
       "DCA_full"          = dca_model,
       "DCA_summary"       = dca_summary),
  "Calibration_DCA.xlsx")
cat("✓ Xuất Calibration_DCA.xlsx\n")