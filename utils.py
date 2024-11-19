def print_summary(data):
    """
    Prints a summary of the dataset, including descriptive statistics and missing values.
    """
    print("Data Summary:")
    print(data.describe())
    print("Missing Values:", data.isnull().sum())