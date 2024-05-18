### 32. `diff` method

The `diff` method accepts a single parameter `n` and takes the difference between the current row and the `n` previous
row. For instance, if a column has the values [5, 10, 2] and `n=1`, the `diff` method would return [NaN, 5, -8]. The
first value is missing because there is no value preceding it.

The `diff` method is a non-aggregating method as well, but there is no direct numpy function that computes it. Instead,
we will define a function within this method that computes this difference.

Complete the body of the `func` function.

Allow `n` to be either a negative or positive integer. You will have to set the first or last n values to `np.nan`. If
you are doing this on an integer column, you will have to convert it to a float first as integer arrays cannot contain
missing values. Use `np.roll` to help shift the data in the arrays.

Test with `test_diff`

### 33. `pct_change` method

The `pct_change` method is nearly identical to the `diff` method. The only difference is that this method returns the
percentage change between the values and not the raw difference. Again, complete the body of the `func` function.

Test with `test_pct_change`

### 34. Arithmetic and Comparison Operators

All the common arithmetic and comparison operators will be made available to our DataFrame. For example, `df + 5` uses
the plus operator to add 5 to each element of the DataFrame. Take a look at some of the following examples:

```python
df + 5
df - 5
df > 5
df != 5
5 + df
5 < df
```

All the arithmetic and comparison operators have corresponding special methods that are called whenever the operator is
used. For instance `__add__` is called when the plus operator is used, and `__le__` is called whenever the less than or
equal to operator is used. See [the full list][14] in the documentation.

Each of these methods accepts a single parameter, which we have named `other`. All of these methods call a more
generic `_oper` method which you will complete.

Within the `_oper` method check if `other` is a DataFrame. Raise a `ValueError` if this DataFrame not one column.
Otherwise, reassign `other` to be a 1D array of the values of its only column.

If `other` is not a DataFrame do nothing and continue executing the rest of the method. We will not check directly if
the types are compatible. Instead we will pass this task onto numpy. So, `df + 5` should work if all the columns in `df`
are booleans, integers, or floats.

Iterate through all the columns of your DataFrame and apply the operation to each array. You will need to use
the `getattr` function along with the `op` string to retrieve the underlying numpy array method. For
instance, `getattr(values, '__add__')` returns the method that uses the plus operator for the numpy array `values`.
Return a new DataFrame with the operation applied to each column.

Run all the tests in class `TestOperators`

### 35. `sort_values` method

This method will sort the rows of the DataFrame by one or more columns. Allow the parameter `by` to be either a single
column name as a string or a list of column names as strings. The DataFrame will be sorted by this column or columns.

The second parameter, `asc`, will be a boolean controlling the direction of the sort. It is defaulted to `True`
indicating that sorting will be ascending  (lowest to greatest). Raise a `TypeError` if `by` is not a string or list.

You will need to use numpy's `argsort` to get the order of the sort for a single column and `lexsort` to sort multiple
columns.

Run the following tests in the `TestMoreMethods` class.

* `test_sort_values`
* `test_sort_values_desc`
* `test_sort_values_two`
* `test_sort_values_two_desc`

### 36. `sample` method

This method randomly samples the rows of the DataFrame. You can either choose an exact number to sample with `n` or a
fraction with `frac`. Sample with replacement by using the boolean `replace`. The `seed` parameter will be used to set
the random number seed.

Raise a `ValueError` if `frac` is not positive and a `TypeError` if `n` is not an integer.

You will be using numpy's random module to complete this method. Within it are the `seed` and `choice` functions. The
latter function has a `replace` parameter that you will need to use. Return a new DataFrame with the new random rows.

Run `test_sample` to test.

### 37. `pivot_table` method

This is a complex method to implement. This method allows you to create a [pivot table][5] from your DataFrame. The
following image shows the final result of calling the pivot table on a DataFrame. It summarizes the mean salary of each
gender for each race.

![pt][6]

A typical pivot table uses two columns as the **grouping columns** from your original DataFrame. The unique values of
one of the grouping columns form a new column in the new DataFrame. In the example above, the race column had five
unique values.

The unique values of the other grouping column now form the columns of the new DataFrame. In the above example, there
were two unique values of gender.

In addition to the grouping columns is the **aggregating column**. This is typically a numeric column that will get
summarized. In the above pivot table, the salary column was aggregated.

The last component of a pivot table is the **aggregating function**. This determines how the aggregating columns get
aggregated. Here, we used the `mean` function.

The syntax used to produce the pivot table above is as follows:

```python
df.pivot_table(rows='race', columns='gender', values='salary', aggfunc='mean')
```

`rows` and `columns` will be assigned the grouping columns. `values` will be assigned the aggregating column
and `aggfunc` will be assigned the aggregating function. All four parameters will be strings. Since `aggfunc` is a
string, you will need to use the builtin `getattr` function to get the correct numpy function.

There are several approaches that you can take to implement this. One approach involves using a dictionary to store the
unique combinations of the grouping columns as the keys and a list to store the values of the aggregative column. You
could iterate over every single row and then use a two-item tuple to hold the values of the two grouping columns.
A `defaultdict` from the collections module can help make this easier. Your dictionary would look something like this
after you have iterated through the data.

```python
{('black', 'male'): [50000, 90000, 40000],
 ('black', 'female'): [100000, 40000, 30000]}
 ```

Once you have mapped the groups to their respective values, you would need to iterate through this dictionary and apply
the aggregation function to the values. Create a new dictionary for this.

From here, you need to figure out how to turn this dictionary into the final DataFrame. You have all the values, you
just need to create a dictionary of columns mapped to values. Use the first column as the unique values of the rows
column.

Other features:

* Return a DataFrame that has the rows and columns sorted
* You must make your pivot table work when passed just one of `rows` or `columns`. If just `rows` is passed return a
  two-column DataFrame with the first column containing the unique values of the rows and the second column containing
  the aggregations. Title the second column the same name as `aggfunc`.
* If `aggfunc` is `None` and `values` is not None then raise a `ValueError`.
* If `values` is `None` and `aggfunc` is not then raise a `ValueError` as there are no values to be aggregated.
* If `aggfunc` and `values` are both `None` then set `aggfunc` equal to the string 'size'. This will produce a
  contingency table (the raw frequency of occurrence). You might need to create an empty numpy array to be a placeholder
  for the values.

Run `test_pivot_table_rows_or_cols` and `test_pivot_table_both` in the `TestGrouping` class.

### 38. Automatically add documentation

All docstrings can be retrieved programmitcally with the `__doc__` special attribute. Docstrings can also be dynamically
set by assigning this same special attribute a string.

This method is already completed and automatically adds documentation to the aggregation methods by setting
the `__doc__` special attribute.

### 39. String-only methods with the `str` accessor

Look back up at the `__init__` method. One of the last lines defines `str` as an instance variable assigned to a new
instance of `StringMethods`. Pandas uses the same variable name for its DataFrames and calls it a string 'accessor'. We
will also refer to it as an accessor as it gives us access to string-only methods.

Scroll down below the definition of the `DataFrame` class. You will see the `StringMethods` class defined there. During
initialization it stores a reference to the underlying DataFrame with `_df`.

There are many string methods defined in this class. The first parameter to each string method is the name of the column
you would like to apply the string method to. We will only allow our accessor to work on a single column of the
DataFrame.

You will only be modifying the `_str_method` which accepts the string method, the name of the column, and any extra
arguments.

Within `_str_method` select the underlying numpy array of the given `col`. Raise a `TypeError` if it does not have
kind 'O'.

Iterate over each value in the array and pass it to `method`. It will look like this: `method(val, *args)`. Return a
one-column DataFrame with the new data.

Test with class `TestStrings`

### 40. Reading simple CSVs

It is important that our library be able to turn data in files into DataFrames. The `read_csv` function, at the very end
of our module, will read in simple comma-separated value files (CSVs) and return a DataFrame.

The `read_csv` function accepts a single parameter, `fn`, which is a string of the file name containing the data. Read
through each line of the file. Assume the values in each line are separated by commas. Also assume the first line
contains the column names.

Create a dictionary to hold the data and return a new DataFrame. Use the file `employee.csv` in the `data` directory to
test your function manually.

Run all the tests in the `TestReadCSV` class.
