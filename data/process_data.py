import sys
import os
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.concat([messages, categories], axis=1, join='inner')

    # create a dataframe of the 36 individual category columns
    categories = categories.categories.str.split(';', expand=True)
    
    # select the first row of the categories dataframe
    row = categories.iloc[0]
    
    # use this row to extract a list of new column names for categories.
    category_colnames = [ x.split('-')[0] for x in row ]
    categories.columns = category_colnames
    
    # Convert category values to just numbers 0 or 1.¶
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str.split('-', expand=True)[1]
        
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    
    # drop the original categories column from `df`
    df = df.drop('categories', axis=1)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    return df, category_colnames


def clean_data(df, category_colnames):
    # drop duplicates
    df.drop_duplicates(inplace=True)
    
    # remove NA values from the categories columns as they will be predicted values
    df.dropna(subset=category_colnames, how='any', inplace=True)
    
    # all categories should have values of 0 or 1; remove the odd 2 value as mentioned here:
    # https://knowledge.udacity.com/questions/64417
    df = df[df['related'] != 2] 
    return df


def save_data(df, database_filename):
    try:
        os.remove(database_filename)
    except OSError:
        pass
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('dr', engine, index=False)
    return

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df, category_colnames = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df, category_colnames)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()