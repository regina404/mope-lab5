import random
import numpy as np
from scipy.stats import f, t
import sklearn.linear_model as lm
from time import process_time


def main(m_tmp):
    m = m_tmp
    n = 15
    x1min = -10
    x1max = 1
    x2min = -9
    x2max = 7
    x3min = -8
    x3max = 3

    ymax = 200 + (x1max + x2max + x3max) / 3
    ymin = 200 + (x1min + x2min + x3min) / 3

    l = 1.215

    # Матриця ПФЕ
    xn = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
          [-1, -1, -1, -1, 1, 1, 1, 1, -l, l, 0, 0, 0, 0, 0],
          [-1, -1, 1, 1, -1, -1, 1, 1, 0, 0, -l, l, 0, 0, 0],
          [-1, 1, -1, 1, -1, 1, -1, 1, 0, 0, 0, 0, -l, l, 0]]

    x1x2_norm, x1x3_norm, x2x3_norm, x1x2x3_norm, x1kv_norm, x2kv_norm, x3kv_norm = [0] * n, [0] * n, [0] * n, [0] * n, [0] * n, [0] * n, [0] * n

    for i in range(n):
        x1x2_norm[i] = xn[1][i] * xn[2][i]
        x1x3_norm[i] = xn[1][i] * xn[3][i]
        x2x3_norm[i] = xn[2][i] * xn[3][i]
        x1x2x3_norm[i] = xn[1][i] * xn[2][i] * xn[3][i]
        x1kv_norm[i] = round(xn[1][i] ** 2, 3)
        x2kv_norm[i] = round(xn[2][i] ** 2, 3)
        x3kv_norm[i] = round(xn[3][i] ** 2, 3)

    Y_matrix = [[random.randint(int(ymin), int(ymax)) for i in range(m)] for j in range(n)]

    print("Матриця планування y:")
    for i in range(15):
        print(Y_matrix[i])

    x01 = (x1max + x1min) / 2
    x02 = (x2max + x2min) / 2
    x03 = (x3max + x3min) / 2

    delta_x1 = x1max - x01
    delta_x2 = x2max - x02
    delta_x3 = x3max - x03

    x0 = [1] * n
    x1 = [-4, -4, -4, -4, 4, 4, 4, 4, -l * delta_x1 + x01, l * delta_x1 + x01, x01, x01, x01, x01, x01]
    x2 = [-10, -10, 4, 4, -10, -10, 4, 4, x02, x02, -l * delta_x2 + x02, l * delta_x2 + x02, x02, x02, x02]
    x3 = [-5, 6, -5, 6, -5, 6, -5, 6, x03, x03, x03, x03, -l * delta_x3 + x03, l * delta_x3 + x03, x03]

    x1x2, x1x3, x2x3, x1x2x3 = [0] * n, [0] * n, [0] * n, [0] * n

    x1kv, x2kv, x3kv = [0] * 15, [0] * 15, [0] * 15

    for i in range(n):
        x1x2[i] = round(x1[i] * x2[i], 3)
        x1x3[i] = round(x1[i] * x3[i], 3)
        x2x3[i] = round(x2[i] * x3[i], 3)
        x1x2x3[i] = round(x1[i] * x2[i] * x3[i], 3)
        x1kv[i] = round(x1[i] ** 2, 3)
        x2kv[i] = round(x2[i] ** 2, 3)
        x3kv[i] = round(x3[i] ** 2, 3)

    y_average = []
    for i in range(len(Y_matrix)):
        y_average.append(np.mean(Y_matrix[i], axis=0))
        y_average = [round(i, 3) for i in y_average]

    list_for_b = list(zip(xn[0], xn[1], xn[2], xn[3], x1x2_norm, x1x3_norm, x2x3_norm, x1x2x3_norm, x1kv_norm,
                          x2kv_norm, x3kv_norm))
    list_for_a = list(zip(x0, x1, x2, x3, x1x2, x1x3, x2x3, x1x2x3, x1kv, x2kv, x3kv))

    print("\nМатриця планування з нормованими коефіцієнтами X:")
    for i in range(15):
        print(list_for_b[i])

    skm = lm.LinearRegression(fit_intercept=False)
    skm.fit(list_for_b, y_average)
    b = skm.coef_
    b = [round(i, 3) for i in b]

    print("\nРівняння регресії зі знайденими коефіцієнтами: \n" "y = {} + {}*x1 + {}*x2 + {}*x3 + {}*x1x2 + {}*x1x3 +"
          " {}*x2x3 + {}*x1x2x3 {}*x1^2 + {}*x2^2 + {}*x3^2".format(b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7], b[8],
                                                                    b[9], b[10]))
    t3_start = process_time
    print("\nПЕРЕВІРКА ОДНОРІДНОСТІ ДИСПЕРСІЇ ЗА КРИТЕРІЄМ КОХРЕНА")
    print("Середні значення відгуку за рядками:", "\n", +y_average[0], y_average[1], y_average[2], y_average[3],
          y_average[4], y_average[5], y_average[6], y_average[7])

    dispersions = []
    for i in range(len(Y_matrix)):
        a = 0
        for k in Y_matrix[i]:
            a += (k - np.mean(Y_matrix[i], axis=0)) ** 2
        dispersions.append(a / len(Y_matrix[i]))
    print("\ndispersion: \n", dispersions)

    Gp = max(dispersions) / sum(dispersions)
    Gt = 0.3346

    if Gp > Gt:
        print("Дисперсія неоднорідна")
        m += 1
        main(m)
    else:
        print("Дисперсія однорідна")
    t3_stop = process_time()
    # критерій Стьюдента
    t2_start = process_time()
    print("\nПЕРЕВІРКА ЗНАЧУЩОСТІ КОЕФІЦІЄНТІВ ЗА КРИТЕРІЄМ СТЬЮДЕНТА")
    sb = sum(dispersions) / len(dispersions)
    sbs = (sb / (m * n)) ** (1/2)

    t_array = [abs(b[i]) / sbs for i in range(0, 11)]

    d = 0
    res = [0] * 11
    coefficients = []
    F3 = (m - 1) * n

    for i in range(n-4):
        if t_array[i] < t.ppf(q=0.975, df=F3):
            res[i] = 0
            print('Виключаємо з рівняння статистично незначущий коефіціент b', i)
        else:
            coefficients.append(b[i])
            res[i] = b[i]
            d += 1

    print("\nЗначущі коефіцієнти регресії:", coefficients)

    y_st = []
    for i in range(n):
        y_st.append(res[0] + res[1] * xn[1][i] + res[2] * xn[2][i] + res[3] * xn[3][i] + res[4] * x1x2_norm[i] \
                    + res[5] * x1x3_norm[i] + res[6] * x2x3_norm[i] + res[7] * x1x2x3_norm[i])
    print("\nЗначення з отриманими коефіцієнтами:\n", y_st)
    t2_stop = process_time()
    t1_start = process_time()

    print("\nПЕРЕВІРКА АДЕКВАТНОСТІ ЗА КРИТЕРІЄМ ФІШЕРА")
    Sad = m * sum([(y_st[i] - y_average[i]) ** 2 for i in range(n)]) / (n - d)
    Fp = Sad / sb
    f4 = n - d
    t1_stop = process_time()
    print("Час ФІШЕРА{},КРИТЕРІЄМ СТЬЮДЕНТА{},КРИТЕРІЄМ КОХРЕНА{}".format(t1_stop-t1_start,t2_stop-t2_start,t3_stop-t3_start))

    if Fp > f.ppf(q=0.95, dfn=f4, dfd=F3):
        print("Рівняння регресії неадекватно оригіналу при рівні значимості 0.05")
    else:
        print("Рівняння регресії адекватно оригіналу при рівні значимості 0.05")


if __name__ == '__main__':
    main(3)
