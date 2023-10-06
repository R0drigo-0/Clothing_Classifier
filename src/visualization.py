import matplotlib.pyplot as plt
from utils_data import *
from utils import *
import numpy as np
import Kmeans
import copy
import time
import KNN

MAX_TOP_N = 20
MAX_IMG = 150
MAX_K = 6

print("Processing data...")


def show_shape(test_imgs, shape_list, query=None, ok=None, title=None):
    res = Retrieval_by_shape(test_imgs, shape_list, query)
    visualize_retrieval(res, MAX_TOP_N, ok=ok, title=str(title))


def show_color(test_imgs, colors_list, query=None, ok=None, title=None):
    res = Retrieval_by_color(test_imgs, colors_list, query)
    visualize_retrieval(res, MAX_TOP_N, ok=ok, title=str(title))


def show_combined(
    test_imgs,
    shape_list,
    colors_list,
    query_shape,
    query_color,
    shape_percentage=None,
    color_percentage=None,
    ok_shape=None,
    ok_color=None,
):
    res = Retrieval_combined(
        test_imgs,
        shape_list,
        colors_list,
        query_shape,
        query_color,
        shape_percentage,
        color_percentage,
    )

    combined_ok = copy.copy(ok_color)
    for index, value in enumerate(ok_shape):
        combined_ok[index].append(value)

    visualize_retrieval(res, MAX_TOP_N, ok=combined_ok)


def show_kmeans_statistics(iterations_list, time_list, wcd_list):
    ax = plt.axes(projection="3d")
    ax.set_title("Kmean statistics")
    ax.set_xlabel("Time")
    ax.set_ylabel("WCD")
    ax.set_zlabel("Iterations")

    x_data = time_list
    y_data = wcd_list
    z_data = iterations_list
    ax.scatter3D(x_data, y_data, z_data)
    plt.show()


def show_kmeans_statistics_2D(param_a, param_b):
    plt.title("Performance (FD)")
    plt.xlabel("Iterations")
    plt.ylabel("Time (sec)")

    plt.plot(param_a, param_b, marker="o")
    plt.annotate(str(param_b[-1]), xy=(param_a[-1], param_b[-1]))

    plt.grid()
    plt.show()


def show_shape_accuracy(knn_labels, gt_labels):
    percentage = Get_shape_accuracy(knn_labels, gt_labels)
    print(percentage)
    return percentage


def show_color_accuracy(kmeans_lables, gt_labels):
    percentage = Get_color_accuracy(kmeans_lables, gt_labels)
    print(percentage)
    return percentage


def show_all_acurracy_tolerance(km, shape_list, test_class_labels, step=5):
    tolerance_list = list(range(10, 100, step))
    acurracy_list = list()

    for i in tolerance_list:
        km.options["tolerance"] = i
        print(km.options["tolerance"])
        acurracy_list.append(show_shape_accuracy(shape_list, test_class_labels))

    print(acurracy_list)
    return acurracy_list


def show_ratio_accuracy_k_shape(max_k, alt_dist):
    dist = "Euclidean"
    if alt_dist == True:
        dist = "Manhattan"

    plt.title(f"Ratio Accuracy/K (Distance Algorithm:{dist})")
    plt.xlabel("K")
    plt.ylabel("Accuracy")

    ratio_list = Get_ratio_accuracy_k_shape(max_k)
    param_a = list(range(2, len(ratio_list) + 2))
    param_b = ratio_list

    plt.plot(param_a, param_b, marker="o")
    plt.annotate(str(param_b[0]), xy=(param_a[0], param_b[0]))
    plt.annotate(str(param_b[-1]), xy=(param_a[-1], param_b[-1]))

    plt.grid()
    plt.show()


if __name__ == "__main__":
    # Load all the images and GT
    (
        train_imgs,
        train_class_labels,
        train_color_labels,
        test_imgs,
        test_class_labels,
        test_color_labels,
    ) = read_dataset(root_folder="./images/", gt_json="./images/gt.json")

    # List with all the existent classes
    classes = list(set(list(train_class_labels) + list(test_class_labels)))

    # Load extended ground truth
    imgs, class_labels, color_labels, upper, lower, background = read_extended_dataset()
    cropped_images = crop_images(imgs, upper, lower)

    # Extended images (Not by default)
    (
        e_imgs,
        e_class_labels,
        e_color_labels,
        e_upper,
        e_lower,
        e_background,
    ) = read_extended_dataset()

    def Retrieval_by_color(imgs, colors_labels, query_color, color_percentages=None):
        matching_images = list()

        for index, value in enumerate(colors_labels):
            if query_color in value:
                if index < len(imgs):
                    matching_images.append(imgs[index])

        """
        matching_images.sort(
            key=lambda x: sum(
                color_percentages[i] for i in x[1] if i in color_percentages
            ),
            reverse=True,
        )
        """

        return np.array(matching_images)

    def Retrieval_by_shape(imgs, class_labels, query_shape, neighbor_percentages=None):
        matching_images = []

        for index, value in enumerate(class_labels):
            if query_shape in value:
                if index < len(imgs):
                    matching_images.append(imgs[index])

        """
        matching_images.sort(
            key=lambda x: sum(
                color_percentages[i] for i in x[1] if i in color_percentages
            ),
            reverse=True,
        )
        """

        return np.array(matching_images)

    def Retrieval_combined(
        imgs,
        class_labels,
        color_labels,
        query_shape,
        query_color,
        shape_percentages=None,
        color_percentages=None,
    ):
        matching_images = list()
        combined_class_color_lables = list()
        for i in range(len(color_labels)):
            print(i)
            aux_class_labels = np.array([class_labels[i]])
            aux_color_labels = np.array(color_labels[i])

            combined_class_color_lables.append(
                list([aux_class_labels, aux_color_labels])
            )

        for index, value in enumerate(combined_class_color_lables):
            if query_shape in value[0] and query_color in value[1]:
                if index < len(imgs):
                    matching_images.append(imgs[index])

        return np.array(matching_images)

    def Get_shape_accuracy(knn_labels, gt_labels):
        n_knn_labels = len(knn_labels)

        correct_labels = 0
        for i in knn_labels:
            if i in gt_labels:
                correct_labels += 1

        accuracy = (correct_labels / n_knn_labels) * 100
        return accuracy

    def Get_color_accuracy(kmeans_lables, gt_labels):
        n_kmeans_lables = len(kmeans_lables)

        correct_labels = 0
        for i in kmeans_lables:
            # i[1:] remove white background
            aux = i[1:]
            aux = [*set(aux)]  # Only unique elements
            cont = 0
            search = True
            while cont < len(gt_labels) and search:
                if aux == gt_labels[cont]:
                    correct_labels += 1
                    search = False
                cont += 1
        accuracy = (correct_labels / n_kmeans_lables) * 100
        return accuracy

    def Kmean_statistics(kmeans_list, k_max):
        km = None

        iterations_list = list()
        time_list = list()
        wcd_list = list()

        for i in range(len(kmeans_list)):
            km = kmeans_list[i]

            for i in range(2, k_max, 1):
                km.K = i

                start = time.time()
                km.fit()
                end = time.time()

                delta = end - start

                wcd = km.withinClassDistance()

                wcd_list.append(wcd)
                time_list.append(delta)
                iterations_list.append(i)

        return iterations_list, time_list, wcd_list

    def Get_ratio_accuracy_k_color(max_k, options=None):
        res = list()
        for k in range(max_k):
            for index, value in enumerate(train_imgs[:MAX_IMG]):
                color_list = list()
                km = Kmeans.KMeans(value)
                for j in range(2, max_k + 1):
                    km.fit()

                color_list.append(list(list(Kmeans.get_colors(km.centroids))))

            print(
                f"Color predicted: {color_list[-1]} Color: {train_color_labels[index]}"
            )
            res.append(Get_color_accuracy(color_list, train_color_labels))

        return res

    def Get_ratio_accuracy_k_shape(max_k):
        res = list()
        for k in range(1, max_k + 1, 1):
            knn = KNN.KNN(train_imgs[:MAX_IMG], train_class_labels)  # Train model
            shape_list = knn.predict(test_imgs, k + 1)  # Predict classes
            res.append(Get_shape_accuracy(shape_list, train_class_labels))
        return res

    # Get predicted clases

    def predicted_clases():
        knn = KNN.KNN(train_imgs[:MAX_IMG], train_class_labels)  # Train model
        shape_list = knn.predict(test_imgs, MAX_K)  # Predict classes
        return knn, shape_list

    knn, shape_list = predicted_clases()

    # Get predicted colors
    def predicted_color():
        color_list = list()
        for i in test_imgs[:MAX_IMG]:
            km = Kmeans.KMeans(i)
            km.find_bestK(MAX_K)
            color_list.append(list(list(Kmeans.get_colors(km.centroids))))
        return km, color_list

    km, color_list = predicted_color()
    knn, shape_list = predicted_clases()

    #######################################################################
    #                              Plots                                  #
    #######################################################################

    """
    show_shape_accuracy(shape_list, test_class_labels)
    """

    """
    show_color_accuracy(color_list, test_color_labels)
    """

    """
    show_all_acurracy_tolerance(km, shape_list, test_class_labels)
    """

    """
    print(Get_ratio_accuracy_k_color(MAX_K))
    """

    # print(Get_ratio_accuracy_k_color(MAX_K))  # KMEANS


def quality_test(color_test_list=[], shape_test_list=[], combined_test_list=[]):
    """
    Displays images that have been automatically selected based on a search parameter
    #####################################################
        To view the next plot, close the current one     
    ##################################################### 
    
    :param color_test_list: List of colors to be displayed on a graph; Format: ['Red', ..., 'Blue']
    :param shape_test_list: List of classes to be displayed in a graph; Format: ['Jeans', ..., 'Dresses']
    :param combined_test_list: List of classes and colors to be displayed in a graph; Format: [['Jeans','Blue'], ..., ['Dresses','Red]]
    """
    for i in color_test_list:
        show_color(test_imgs, color_list, i, title=f"{i} Color", ok=test_color_labels)

    for i in shape_test_list:
        show_shape(test_imgs, shape_list, i, title=f"{i} Shape", ok=test_class_labels)


def quantity_test(k=MAX_K):
    """
    Shows data and graphs about the execution of the program.
    #####################################################
        To view the next plot, close the current one     
    ##################################################### 

    :param k: Parameter that can be added if you want to use a different value than the default
    """

    iteration_list, time_list, wcd_list = Kmean_statistics([km], k)
    show_kmeans_statistics(iteration_list, time_list, wcd_list)
    show_kmeans_statistics_2D(iteration_list, time_list)

    show_ratio_accuracy_k_shape(MAX_K, knn.alt_dist)

    Plot3DCloud(km)
    plt.show()


quality_test(["Blue"], ["Shorts"])
# quantity_test(100)
