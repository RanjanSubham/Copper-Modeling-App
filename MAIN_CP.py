import streamlit as st
import pandas as pd
import numpy as np
import pickle
import pyodbc
import matplotlib.pyplot as plt 
import scipy.stats as stats


# Set the title and layout of the app

# Load pre-trained models and scalers
with open('model.pkl', 'rb') as file:
    regressor_model = pickle.load(file)
with open('scaler.pkl', 'rb') as file:
    scaler_regressor = pickle.load(file)
with open('t.pkl', 'rb') as file:
    ohe_regressor = pickle.load(file)
with open('cmodel.pkl', 'rb') as file:
    classifier_model = pickle.load(file)
with open('cscaler.pkl', 'rb') as file:
    scaler_classifier = pickle.load(file)
with open('ct.pkl', 'rb') as file:
    ohe_classifier = pickle.load(file)
with open('s.pkl', 'rb') as file:
    ohe_status_classifier = pickle.load(file)
    
    

# Fetch data from SQL database
def fetch_data(query):
    try:
        conn = pyodbc.connect(connection_string)
        cursor = conn.cursor()
        cursor.execute(query)
        data = cursor.fetchall()
        
        return data
    except Exception as e:
        st.error(f"Error: {e}")
        return None

# Load connection parameters from pickle file
with open('sql_data.pkl', 'rb') as f:
    connection_string, _ = pickle.load(f)

# Load all .pkl files for displaying data
def load_pkl_data(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

country_sales_data = load_pkl_data('country_sales.pkl')
app_avg_qty_data = load_pkl_data('application_avg_qty.pkl')
item_type_win_count_data = load_pkl_data('item_type_win_count.pkl')
top_bottom_customers_data = load_pkl_data('top_bottom_customers.pkl')
country_sales_2_data = load_pkl_data('country_sales_2.pkl')

# Function to predict status
def predict_status(quantity_tons, application, thickness, width, country, customer, product_ref, item_type):
    try:
        new_sample = np.array([[np.log(float(quantity_tons)), float(application), np.log(float(thickness)),
                                float(width), float(country), float(customer), float(product_ref), item_type]])
        new_sample_ohe_item = ohe_classifier.transform(new_sample[:, [7]]).toarray()
        new_sample = np.concatenate((new_sample[:, [0, 1, 2, 3, 4, 5, 6]], new_sample_ohe_item), axis=1)
        new_sample = scaler_classifier.transform(new_sample)
        status_prediction = classifier_model.predict(new_sample)
        return status_prediction[0]
    except Exception as e:
        st.error(f"Error in prediction: {e}")
        return None



# Set background image for the entire app
st.markdown(
    """
    <style>
    .stApp {
        background-image: url('https://images.freecreatives.com/wp-content/uploads/2016/04/Free-Website-BAckgrounds1.jpg');
        background-size: cover;
        background-position: center center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    .stMarkdown, .stText, .stButton, .stTextInput, .stTextArea, .stSelectbox, .stRadio, .stSidebar, .stTabs, .stTable {
        font-weight: bold;
    }
    .stTabs .stTab, .stTabs .stTabContent {
        font-weight: bold;
        font-size: 1.2em;
    }
    .content-wrapper {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100vh;
        text-align: center;
        padding: 20px;
    }
    .content-wrapper h1 {
        margin: 0;
        font-size: 4em;
        color: #ffffff;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.6);
        letter-spacing: 1px;
        line-height: 1.2;
        text-transform: uppercase;
    }
    .stButton>button {
        background-color: #ff4b4b;
        color: white;
        border: none;
        padding: 10px 20px;
        font-size: 1em;
        border-radius: 5px;
    }
    .stButton>button:hover {
        background-color: #ff1a1a;
    }
    .stSidebar {
        background-color: #333;
        color: white;
    }
    .stSelectbox>div>div>button, .stTextInput>div>div>input, .stTextArea>div>div>textarea {
        background-color: #444;
        color: white;
        border: none;
    }
    .stRadio>div>div>label {
        color: white;
    }
    .stMarkdown h2 {
        color: #00c8ff;
    }
    .stMarkdown h4 {
        color: white;
    }
    .footer {
        text-align: center;
        margin-top: 50px;
        color: #00c8ff;
    }
    .footer a {
        color: #00c8ff;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Custom content wrapper with a heading
st.markdown(
    """
    <div class="content-wrapper">
        <h1>Welcome to Industrial Copper Modeling</h1>
    </div>
    """,
    unsafe_allow_html=True
)


# Add a central welcome message









# Streamlit App

# Tabs
tab1, tab2, tab3, tab4, tab5,tab6,tab7,tab8,tab9 = st.tabs(["About Project","Column Details","Statistics","PREDICT SELLING PRICE", "PREDICT STATUS", "SQL Query", "Data Analysis", "Random Values","Developer Details"])

tab1.markdown(
    """
    <style>
    .tab1-content {
        background-image: url('https://images.freecreatives.com/wp-content/uploads/2016/04/Free-Website-BAckgrounds1.jpg');
        background-size: cover;
        background-position: center center;
        background-repeat: no-repeat;
        background-attachment: fixed;
        padding: 20px;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

with tab1:
    st.markdown(
        """
        <style>
        .tab1-background {
            background-image: url('https://images.freecreatives.com/wp-content/uploads/2016/04/Solid-Black-Website-Background.jpg');
            background-size: cover;
            background-position: center center;
            background-repeat: no-repeat;
            background-attachment: fixed;
            height: 100vh; /* Full height of the viewport */
            color: white;
            padding: 20px;
        }
        .tab1-content-wrapper {
            background-color: rgba(0, 0, 0, 0.5); /* Semi-transparent background for better readability */
            padding: 20px;
            border-radius: 10px;
        }
        </style>
        <div class="tab1-background">
            <div class="tab1-content-wrapper">
                <h1>Welcome to Industrial Copper Modeling</h1>
                <h4>About the project</h4>
                <p>
                The copper industry deals with data related to sales and pricing. However, this data may suffer from issues such as skewness and noisy data, 
                which can affect the accuracy of manual predictions. Dealing with these challenges manually can be time-consuming and may not result in optimal 
                pricing decisions. A machine learning regression model can address these issues by utilizing advanced techniques such as data normalization, 
                feature scaling, and outlier detection, and leveraging algorithms that are robust to skewed and noisy data.
                </p>
                <p>
                The steps involved in the project are,
                <ol>
                <li>Explored skewness and fixed the outliers in the dataset</li>
                <li>Transformed the data and performed necessary cleaning and pre-processing steps</li>
                <li>Built the ML Classification model which predicts Status: WON or LOST</li>
                <li>Built the ML Regression model which predicts continuous variable ‘Selling_Price’</li>
                <li>Created a Streamlit page where you can insert each column value and you will get the Selling_Price predicted value or Status (Won/Lost)</li>
                </ol>
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
st.markdown('<div class="footer">Made by Subham Ranjan</div>', unsafe_allow_html=True)     
    
with tab2:
    # Apply the same styling as tab1
   
    # Define column titles
    tab_titles = [
        "Unnamed: 0",
        "id",
        "item_date",
        "quantity tons",
        "customer",
        "country",
        "status",
        "item type",
        "thickness",
        "width",
        "product_ref",
        "selling_price",
        "application"
    ]

    # Define column details
    column_details = {
        "Unnamed: 0": {
            "Type": "Integer",
            "Description": "This column is often a placeholder or an index column added by default when exporting data. It usually holds row numbers or indices but doesn't contain meaningful information beyond indexing the rows.",
        },
        "id": {
            "Type": "String (UUID)",
            "Description": "This is a unique identifier for each record. UUIDs (Universally Unique Identifiers) are used to ensure that each record has a distinct identity, which helps in tracking and managing data without confusion.",
        },
        "item_date": {
            "Type": "Date",
            "Description": "Represents the date when the item was recorded, ordered, or processed. It is crucial for time-based analysis, such as trends over time or seasonality in the data.",
        },
        "quantity tons": {
            "Type": "Float",
            "Description": "Indicates the amount of the item in tons. This numerical value is used for understanding the volume or weight of the item being handled or sold.",
        },
        "customer": {
            "Type": "Integer",
            "Description": "Represents a unique identifier for the customer making the purchase or order. This helps in tracking customer activity and analyzing customer-specific trends.",
        },
        "country": {
            "Type": "Integer",
            "Description": "Represents the country where the transaction took place. This column is used for geographic analysis, such as understanding sales distribution across different regions.",
            "Values": {
                78: "Gambia",
                26: "Kazakhstan",
                27: "Russia",
                32: "Belgium",
                30: "Greece",
                39: "Austria",
                28: "Egypt",
                25: "USA",
                77: "Libya",
                84: "Burkina Faso",
                38: "Switzerland",
                79: "Senegal",
                40: "UK",
                86: "Togo",
                113: "Uganda",
                89: "Liberia",
                107: "Sudan"
            }
        },
        "status": {
            "Type": "String",
            "Description": "Indicates the current status of the order or item. Possible values might include Won (successful), Not lost for AM (status in an approval process), or To be approved (pending approval). It helps in tracking the progress or outcome of orders.",
            "Values": {
                "Lost": "The transaction was unsuccessful, and the deal was lost to a competitor or was otherwise declined.",
                "Won": "The transaction was successful, and the deal was closed with the customer.",
                "Not lost for AM": "The transaction is still under consideration or in a pending state for Account Management and has not been lost to a competitor or declined.",
                "To be approved": "The transaction is pending approval. It may need further review or authorization before it can proceed.",
                "Unknown": "The status of the transaction is not clear or has not been specified. This may need further investigation or clarification.",
                "Draft": "The transaction is in the initial stages and has not been finalized or submitted for approval.",
                "Revised": "The transaction has been modified or updated, possibly after initial submission, to reflect new terms, quantities, prices, etc.",
                "Offered": "An offer has been made to the customer, but it has not yet been accepted or finalized.",
                "Offerable": "The transaction is in a state where it is ready to be offered to the customer, but the actual offer has not been made yet.",
                "Wonderful": "This seems unusual for a status and might be an error or special categorization in your dataset. It would need context from your business process to understand fully."
            }
        },
        "item type": {
            "Type": "String",
            "Description": "Type or category of the item.",
            "Values": {
                "W": "This likely stands for Wire. Copper wire is commonly used in electrical and electronic applications due to its excellent conductivity.",
                "S": "This could represent Sheet or Strip. Copper sheets and strips are used in various industrial applications, including roofing, flashing, and other construction materials, as well as in manufacturing components.",
                "PL": "Likely stands for Plate. Copper plates are thicker and sturdier than sheets, used in industrial machinery, heat exchangers, and other heavy-duty applications.",
                "WI": "This likely stands for Winding Wire or Windings. Used primarily in electrical motors, transformers, and other electromagnets.",
                "Others": "This category includes any copper products not specifically classified into the other categories. It can encompass a variety of specialized or niche products.",
                "IPL": "This could stand for Insulated Power Lines or a similar product involving insulation and power transmission.",
                "SLAWR": "This could stand for a specific type of copper product, possibly Soldering Alloy Wire or a specialized wire type used in industrial applications."
            }
        },
        "application": {
            "Type": "Integer",
            "Description": "Application or use case for the item.",
            "Values": {
                20: "Electrical engineering.",
                65: "Automotive industry.",
                28: "Construction-related.",
                66: "Aerospace or defense.",
                22: "Consumer electronics.",
                26: "Telecommunications.",
                79: "Industrial machinery.",
                58: "Renewable energy.",
                39: "Household appliances.",
                3: "Miscellaneous.",
                67: "Specific industrial applications.",
                68: "Commercial applications.",
                4: "Specialized applications.",
                19: "Additional specific uses.",
                5: "Further industrial uses.",
                70: "Domain-specific.",
                99: "Unique or niche applications.",
                2: "Other applications."
            }
        },
        "thickness": {
            "Type": "Float",
            "Description": "Measurement of the thickness of the item. This is a physical attribute that may influence the item's suitability for different applications or its pricing.",
        },
        "width": {
            "Type": "Float",
            "Description": "Measurement of the width of the item. Like thickness, width is a physical characteristic that affects the item's usability and pricing.",
        },
        "product_ref": {
            "Type": "Integer",
            "Description": "Reference number or identifier for the product. This helps in linking the item to a specific product catalog or inventory system.",
        },
        "selling_price": {
            "Type": "Float",
            "Description": "The price at which the item is sold. This column is crucial for financial analysis, such as calculating total revenue, profit margins, and pricing strategies.",
        }
    }

    # Create a two-column layout
    col1, col2 = st.columns([1, 2])

    with col1:
        selected_tab = st.selectbox("Select a column to view details", tab_titles)

    with col2:
        if selected_tab in column_details:
            details = column_details[selected_tab]
            st.header(f"Details for {selected_tab}")
            st.write(f"**Type:** {details['Type']}")
            st.write(f"**Description:** {details['Description']}")
            if 'Values' in details:
                st.write("**Values:**")
                for key, value in details['Values'].items():
                    st.write(f" - {key}: {value}")
    
      
import seaborn as sns

# Function to load data from CSV
def load_data(file_path):
    return pd.read_csv(file_path)

# Load the data from local file path
file_path = r"C:\Users\Subham Ranjan\Downloads\copper_randomcsv.csv"
df2 = load_data(file_path)

with tab3:
    st.markdown('<div class="header"><h2>Statistical Analysis</h2></div>', unsafe_allow_html=True)
    st.markdown('Enter the details below:')


    # Define tab titles
    tab_titles = [
        "Show Data Information",
        "Show First Few Rows",
        "Show Missing Values",
        "Show Data Shape",
        "Show Column Names",
        "Show Unique Values",
        "Show Correlation Heatmap",
        "Show Point-Biserial Correlation Plot",
        "Descriptive Statistics",
        "Show Columns Consisting Outliers"
    ]

    # Create tabs
    tabs = st.tabs(tab_titles)

    # Show data based on selected tab
    selected_tab = st.radio("Choose a view", tab_titles)
    
    if selected_tab == tab_titles[0]:  # Data Information
        st.subheader('Data Information')
        info_buffer = pd.DataFrame({
            'Column': df2.columns,
            'Dtype': [str(df2[col].dtype) for col in df2.columns],
            'Non-Null Count': [df2[col].count() for col in df2.columns],
            'Null Count': [df2[col].isnull().sum() for col in df2.columns]
        })
        st.write(info_buffer)

    elif selected_tab == tab_titles[1]:  # First Few Rows
        st.subheader('First Few Rows of Data')
        st.write(df2.head())

    elif selected_tab == tab_titles[2]:  # Missing Values
        st.subheader('Missing Values')
        st.write(df2.isnull().sum())

    elif selected_tab == tab_titles[3]:  # Data Shape
        st.subheader('Data Shape')
        st.write(df2.shape)

    elif selected_tab == tab_titles[4]:  # Column Names
        st.subheader('Column Names')
        st.write(df2.columns)

    elif selected_tab == tab_titles[5]:  # Unique Values
        st.subheader('Unique Values')
        st.write("Item Type:", df2["item type"].unique())
        st.write("Country:", df2["country"].unique())
        st.write("Status:", df2["status"].unique())
        st.write("Application:", df2["application"].unique())

    elif selected_tab == tab_titles[6]:  # Correlation Heatmap
        st.subheader('Correlation Heatmap')
        plt.figure(figsize=(10, 8))
        sns.heatmap(df2.corr(numeric_only=True), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title('Correlation Heatmap')
        st.pyplot(plt)

    elif selected_tab == tab_titles[7]:  # Point-Biserial Correlation Plot
        st.subheader('Point-Biserial Correlation Plot')
        
        binary_col = 'status'
        if binary_col in df2.columns:
            df2[binary_col] = df2[binary_col].astype('category').cat.codes
            continuous_cols = df2.select_dtypes(include=['float64', 'int64']).columns

            if binary_col in continuous_cols:
                continuous_cols = continuous_cols.drop(binary_col)

            if continuous_cols.empty:
                st.write("No continuous columns available for correlation calculation.")
            else:
                results = []
                for col in continuous_cols:
                    point_biserial_corr, p_value = stats.pointbiserialr(df2[binary_col], df2[col])
                    results.append((col, point_biserial_corr, p_value))

                results_df = pd.DataFrame(results, columns=['Variable', 'Point-Biserial Correlation', 'P-Value'])

                plt.figure(figsize=(10, 6))
                plt.barh(results_df['Variable'], results_df['Point-Biserial Correlation'], color='skyblue')
                plt.xlabel('Point-Biserial Correlation Coefficient')
                plt.title('Point-Biserial Correlation with Binary Variable')
                plt.axvline(0, color='grey', linestyle='--', linewidth=0.8)
                st.pyplot(plt)
        else:
            st.write(f"Binary column '{binary_col}' not found in the dataset.")

    elif selected_tab == tab_titles[8]:  # Descriptive Statistics
        st.subheader('Descriptive Statistics')
        st.write(df2.describe())

    elif selected_tab == tab_titles[9]:  # Columns Consisting Outliers
        st.subheader('Box Plot of Continuous Variables')
        columns_to_plot = ['thickness', 'quantity tons', 'selling_price']

        plt.figure(figsize=(12, 6))
        sns.boxplot(data=df2[columns_to_plot], palette="Set2")

        plt.title('Box Plot of Continuous Variables')
        plt.xlabel('Variables')
        plt.ylabel('Values')

        st.pyplot(plt)

    
  
with tab4:
    st.markdown('<div class="header"><h2>Predict Selling Price</h2></div>', unsafe_allow_html=True)
    st.markdown('Enter the details below:')
    
    quantity_tons = st.text_input('Quantity Tons  (Min: 1, Max: 1000000000)', key='quantity_tons_price')
    item_type = st.selectbox('Item Type', ['W', 'WI', 'S', 'Others', 'PL', 'IPL', 'SLAWR'], key='item_type_price')
    application = st.selectbox('Application',[10,15,41,42,38,56,59,29,27,25,40,69,20,65,28,66,22,26,79,58,39,3,67,68,4,19,5,70,99,2], key='application_price_s')
    thickness = st.text_input('Thickness (Min: 0.18, Max: 400)', key='thickness_price')
    width = st.text_input('Width (Min: 1, Max: 2990)', key='width_price')
    country = st.selectbox('Country',['78','26','27','32','30','39','28','25','77','84','38','79','40','80','113','89','107'],key='country_price')
    customer = st.text_input('Customer ID (Min: 12458, Max: 30408185)', key='customer_price')
    product_ref = st.text_input('Product Reference (Min: 611728, Max: 1722207579)', key='product_ref_price')
    
    if st.button('Predict', key='predict_button'):
        try:
            new_sample = np.array([[np.log(float(quantity_tons)), float(application), np.log(float(thickness)), 
                                    float(width), float(country), float(customer), float(product_ref), item_type]])
            new_sample_ohe = ohe_regressor.transform(new_sample[:, [7]]).toarray()
            new_sample = np.concatenate((new_sample[:, [0, 1, 2, 3, 4, 5, 6]], new_sample_ohe), axis=1)
            new_sample = scaler_regressor.transform(new_sample)
            new_pred = regressor_model.predict(new_sample)
            st.success(f'Predicted selling price: {np.exp(new_pred[0]):.2f}')
        except Exception as e:
            st.error(f"Error in prediction: {e}")

with tab5:
    st.markdown('<div class="header"><h2>Predict Status</h2></div>', unsafe_allow_html=True)
    st.markdown('Enter the details below:')
    
    quantity_tons = st.text_input('Quantity Tons (Min: 1, Max: 1000000000)', key='quantity_tons_status')
    application = st.selectbox('Application',[10,15,41,42,38,56,59,29,27,25,40,69,20,65,28,66,22,26,79,58,39,3,67,68,4,19,5,70,99,2], key='application_status')
    thickness = st.text_input('Thickness  (Min: 0.18, Max: 400)', key='thickness_status')
    width = st.text_input('Width (Min: 1, Max: 2990)', key='width_status')
    country = st.selectbox('Country',['78','26','27','32','30','39','28','25','77','84','38','79','40','80','113','89','107'],key='country_status')
    customer = st.text_input('Customer ID (Min: 12458, Max: 30408185)', key='customer_status')
    product_ref = st.text_input('Product Reference (Min: 611728, Max: 1722207579)', key='product_ref_status')
    item_type = st.selectbox('Item Type', ['W', 'WI', 'S', 'Others', 'PL', 'IPL', 'SLAWR'], key='item_type_status')
    
    if st.button('Predict', key='status_button'):
        status_prediction = predict_status(quantity_tons, application, thickness, width, country, customer, product_ref, item_type)
        if status_prediction is not None:
            if status_prediction == 1:
                st.success('Predicted Status: Won')
            else:
                st.error('Predicted Status: Lost')

with tab6:
    st.markdown('<div class="header"><h2>SQL Query</h2></div>', unsafe_allow_html=True)
    query = st.text_area('Enter SQL Query:')
    
    if st.button('Execute', key='sql_button'):
        data = fetch_data(query)
        if data:
            st.write(pd.DataFrame(data))
        else:
            st.error('No data fetched. Check your query or connection.')

with tab7:
    st.subheader('Data Analysis')
    st.markdown('Choose an option for data analysis:')
    
    analysis_option = st.selectbox('Select Analysis Option', [
        'The maximum valid quantity-tons ordered in each item-type category.',
        'For which application, the average order quantity is the highest and the lowest?',
        'Which item type has the max number of ‘Win’ status?',
        'N number of customers who are top and bottom contributors in Revenue. User should have an option to choose the value of n. Return order_id and customer_id',
        'Country-wise sum of sales.'
    ])
    
    queries = {
        'The maximum valid quantity-tons ordered in each item-type category.':
            """
            SELECT item_type, MAX(quantity_tons) AS max_quantity_tons 
            FROM Copper 
            WHERE quantity_tons IS NOT NULL 
            GROUP BY item_type;
            """,
        'For which application, the average order quantity is the highest and the lowest?':
            """
            WITH app_qty AS (
                SELECT application, AVG(quantity_tons) AS avg_qty, 
                       ROW_NUMBER() OVER (ORDER BY AVG(quantity_tons) DESC) AS h_rn,
                       ROW_NUMBER() OVER (ORDER BY AVG(quantity_tons) ASC) AS l_rn
                FROM Copper 
                GROUP BY application
            )
            SELECT application, avg_qty 
            FROM app_qty 
            WHERE h_rn = 1 OR l_rn = 1;
            """,
        'Which item type has the max number of ‘Win’ status?':
            """
            select item_type,count(status) as win_cnt from Copper where status = 'Won'
            group by item_type order by count(status) desc
            """,
        'N number of customers who are top and bottom contributors in Revenue. User should have an option to choose the value of n. Return order_id and customer_id':
            """
            DECLARE @N INT = 2;

WITH RevenueRankings AS (
    SELECT 
        customer,
        id AS order_id,
        ROUND(SUM(selling_price), 0) AS revenue,
        ROW_NUMBER() OVER (ORDER BY SUM(selling_price) DESC) AS TopRank,
        ROW_NUMBER() OVER (ORDER BY SUM(selling_price) ASC) AS BottomRank
    FROM Copper 
    GROUP BY customer, id
)
SELECT customer, order_id, revenue, 'Top' AS RankType 
FROM RevenueRankings 
WHERE TopRank <= @N

UNION ALL

SELECT customer, order_id, revenue, 'Bottom' AS RankType 
FROM RevenueRankings 
WHERE BottomRank <= @N

ORDER BY RankType, revenue DESC;

            """,
        'Country-wise sum of sales.':
            """
            SELECT country, SUM(selling_price) AS total_sales 
            FROM Copper 
            GROUP BY country 
            ORDER BY SUM(selling_price) DESC;
            """
    }

    if st.button('Generate Query', key='generate_query_button'):
        query = queries.get(analysis_option)
        st.code(query, language='sql')

    if st.button('Execute Query', key='execute_query_button'):
        if analysis_option == 'The maximum valid quantity-tons ordered in each item-type category.':
            st.write(pd.DataFrame(country_sales_data))
        elif analysis_option == 'For which application, the average order quantity is the highest and the lowest?':
            st.write(pd.DataFrame(app_avg_qty_data))
        elif analysis_option == 'Which item type has the max number of ‘Win’ status?':
            st.write(pd.DataFrame(item_type_win_count_data))
        elif analysis_option == 'N number of customers who are top and bottom contributors in Revenue. User should have an option to choose the value of n. Return order_id and customer_id':
            st.write(pd.DataFrame(top_bottom_customers_data))
        elif analysis_option == 'Country-wise sum of sales.':
            st.write(pd.DataFrame(country_sales_2_data))

with tab8:
    with st.container():  # Use `with` to create a container
        st.markdown('<div class="header"><h2>Random Values</h2></div>', unsafe_allow_html=True)
        
        # Initialize session state for random rows if not already set
        if 'random_rows' not in st.session_state:
            st.session_state.random_rows = df2.sample(n=5)
        
        # Number input for selecting number of random rows
        num_rows = st.number_input(
            'Number of Random Rows', 
            min_value=1, 
            max_value=len(df2), 
            value=5, 
            step=1, 
            key='random_rows_input'
        )
        
        # Button to update random rows
        if st.button('Show Random Rows', key='random_button'):
            st.session_state.random_rows = df2.sample(n=num_rows)
        
        # Display the random rows
        if 'random_rows' in st.session_state:
            st.dataframe(st.session_state.random_rows)
        else:
            st.write("No data to display. Please generate random rows.")

    # Footer
    

    # Footer
    
    



# Content for "Developer Details" tab
with tab9:
    st.title("Developer Details")
    st.subheader("Contact Information")
    st.write("**Mobile No:** +91 7004849473")
    st.write("**LinkedIn:** [Subham Ranjan(https://www.linkedin.com/in/subham-ranjan0525)")
    st.write("**Email:** [subhamranjansbi@gmail.com](mailto:subhamranjansbi@gmail.com)")
    st.write("**GitHub:** [RanjanSubham(https://github.com/RanjanSubham)")

    # Optional: Add some styling
    st.markdown("""
    <style>
        .stApp {
            background-color: #f0f0f5;
        }
        .css-1k4v8ar {
            background-color: #1f1f1f;
        }
        .css-1e3szw1 {
            color: white;
        }
    </style>
    """, unsafe_allow_html=True)
# Implement functionality for tab 9 (Column Details)

        
    
   

# Streamlit app




    
    
    