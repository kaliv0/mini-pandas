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
