import pandas as pd


def split(dataset_path):

    # Load the DataFrame from the pickle file
    df = pd.read_pickle(dataset_path)

    # Filter out the rows where the 'part' column is 'c'
    df_c = df[df['part'] == 'c']

    # Modify the 'file_path' column to replace 'train' with 'val'
    df_c['file_path'] = df_c['file_path'].str.replace('train', 'val')

    # Filter out the rows where the 'part' column is not 'c'
    df_not_c = df[df['part'] != 'c']

    # Save the filtered DataFrames to new pickle files
    df_c.to_pickle('/MagnaTagATune/annotations/validation.pkl')
    df_not_c.to_pickle(dataset_path)

    # Print details of DataFrame where 'part' is 'c'
    print("Details of DataFrame where 'part' is 'c':")
    print(df_c.head())
    print(df_c.tail())
    print(df_c.info())
    print(df_c.describe())

    # Print details of DataFrame where 'part' is not 'c'
    print("\nDetails of DataFrame where 'part' is not 'c':")
    print(df_not_c.head())
    print(df_not_c.tail())
    print(df_not_c.info())
    print(df_not_c.describe())

    # Optionally, save the split DataFrames
    #df_c.to_pickle('train.pkl')
    #df_not_c.to_pickle('val.pkl')
    
    
    
    #return df
    #return df_c, df_not_c


val = split('/MagnaTagATune/annotations/train_labels.pkl')



#print(pd.read_pickle('/mnt/storage/scratch/gv20319/MagnaTagATune/annotations/validation.pkl'))