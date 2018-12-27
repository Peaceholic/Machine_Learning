import matplotlib.pyplot as plt
import mglearn

# Create data set
X, y = mglearn.datasets.make_forge()

# draw scatter plot
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.legend(["Class 0", "Class 1"], loc=4)
plt.xlabel("1st attribute")
plt.ylabel("2nd attribute")
print("X.shape: {}".format(X.shape))

plt.show()
