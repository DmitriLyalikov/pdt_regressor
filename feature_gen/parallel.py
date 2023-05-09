import multiprocessing


def my_function(x):
    return x ** 2

if __name__ == "__main__":
    with multiprocessing.Pool() as p:
        results = p.map(my_function, [1, 2, 3, 4, 5])

    print(results)