
from collections import defaultdict
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score


def rmse(predictions, verbose=True):
    """Compute RMSE (Root Mean Squared Error).
    """

    if not predictions:
        raise ValueError("Prediction list is empty.")

    mse = np.mean(
        [float((true_r - est) ** 2) for (_, _, true_r, est, _) in predictions]
    )
    rmse_ = np.sqrt(mse)

    if verbose:
        print(f"RMSE: {rmse_:1.4f}")

    return rmse_


def mse(predictions, verbose=True):
    """Compute MSE (Mean Squared Error).
    """

    if not predictions:
        raise ValueError("Prediction list is empty.")

    mse_ = np.mean(
        [float((true_r - est) ** 2) for (_, _, true_r, est, _) in predictions]
    )

    if verbose:
        print(f"MSE: {mse_:1.4f}")

    return mse_


def mae(predictions, verbose=True):
    """Compute MAE (Mean Absolute Error).
    """

    if not predictions:
        raise ValueError("Prediction list is empty.")

    mae_ = np.mean([float(abs(true_r - est)) for (_, _, true_r, est, _) in predictions])

    if verbose:
        print(f"MAE:  {mae_:1.4f}")

    return mae_


from collections import defaultdict
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
def another_measures(predictions, N=2, threshold=4, verbose=True):
    # فیلتر کردن اقلام با امتیاز واقعی صفر (اسپارس)
    predictions = [(uid, iid, true_r, est, extra) for uid, iid, true_r, est, extra in predictions if true_r > 0]
    
    if not predictions:
        raise ValueError("Prediction list is empty.")
    
    # گروه‌بندی پیش‌بینی‌ها بر اساس کاربر
    user_predictions = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        user_predictions[uid].append((true_r, est, iid))
    
    y_true_all, y_pred_all = [], []
    for uid, preds in user_predictions.items():
        # مرتب‌سازی پیش‌بینی‌ها بر اساس امتیاز پیش‌بینی‌شده (نزولی) و انتخاب N قلم برتر
        preds_sorted = sorted(preds, key=lambda x: x[1], reverse=True)[:N]
        y_true_user = [float(true_r) for true_r, _, _ in preds_sorted]
        y_pred_user = [float(est) for _, est, _ in preds_sorted]
        y_true_all.extend(y_true_user)
        y_pred_all.extend(y_pred_user)
    
    # تبدیل به مقادیر باینری با آستانه
    y_true_binary = (np.array(y_true_all) >= threshold).astype(int)
    y_pred_binary = (np.array(y_pred_all) >= threshold).astype(int)
    
    # محاسبه معیارها
    precision = precision_score(y_true_binary, y_pred_binary)
    recall = recall_score(y_true_binary, y_pred_binary)
    f1 = f1_score(y_true_binary, y_pred_binary)
    
    if verbose:
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    
    return (precision, recall, f1)
# def another_measures(predictions, N=10, threshold=4, verbose=True):
#     if not predictions:
#         raise ValueError("Prediction list is empty.")
    
#     # گروه‌بندی پیش‌بینی‌ها بر اساس کاربر
#     user_predictions = defaultdict(list)
#     for uid, iid, true_r, est, _ in predictions:
#         user_predictions[uid].append((true_r, est))
    
#     y_true_all, y_pred_all = [], []
#     for uid, preds in user_predictions.items():
#         # مرتب‌سازی پیش‌بینی‌ها بر اساس امتیاز پیش‌بینی‌شده (نزولی)
#         preds_sorted = sorted(preds, key=lambda x: x[1], reverse=True)[:N]
#         y_true_user = [float(true_r) for true_r, _ in preds_sorted]
#         y_pred_user = [float(est) for _, est in preds_sorted]
#         y_true_all.extend(y_true_user)
#         y_pred_all.extend(y_pred_user)
    
#     y_true_binary = (np.array(y_true_all) >= threshold).astype(int)
#     y_pred_binary = (np.array(y_pred_all) >= threshold).astype(int)
    
#     precision = precision_score(y_true_binary, y_pred_binary)
#     recall = recall_score(y_true_binary, y_pred_binary)
#     f1 = f1_score(y_true_binary, y_pred_binary)
    
#     if verbose:
#         print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    
#     return (precision, recall, f1)
# def another_measures(predictions, verbose=True):
#     """Compute Precision, recall, f1 .
#     """
#     if not predictions:
#         raise ValueError("Prediction list is empty.")
    
#     y_true = np.array([float(true_r) for (_, _, true_r, _, _) in predictions])
#     y_pred = np.array([float(est) for (_, _, _, est, _) in predictions])

#     threshold = 4
#     y_true_binary = (y_true >= threshold).astype(int)
#     y_pred_binary = (y_pred >= threshold).astype(int)

#     precision = precision_score(y_true_binary, y_pred_binary)
#     recall = recall_score(y_true_binary, y_pred_binary)
#     f1 = f1_score(y_true_binary, y_pred_binary)

#     return (precision,recall,f1)