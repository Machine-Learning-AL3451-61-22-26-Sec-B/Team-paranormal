import streamlit as st
import pandas as pd

# Function to load data
def load_data(uploaded_file):
    return pd.read_csv(uploaded_file)


# Function for Candidate Elimination Algorithm (CEA)
def learn(concepts, target):
    specific_h = concepts[0].copy()
    general_h = [["?" for i in range(len(specific_h))] for i in range(len(specific_h))]

    for i, h in enumerate(concepts):
        if target[i] == "Yes":
            for x in range(len(specific_h)):
                if h[x] != specific_h[x]:
                    specific_h[x] = '?'
                    general_h[x][x] = '?'

        if target[i] == "No":
            for x in range(len(specific_h)):
                if h[x] != specific_h[x]:
                    general_h[x][x] = specific_h[x]
                else:
                    general_h[x][x] = '?'

    indices = [i for i, val in enumerate(general_h) if val == ['?', '?', '?', '?', '?', '?']]
    for i in indices:
        general_h.remove(['?', '?', '?', '?', '?', '?'])
    
    return specific_h, general_h

# Define Streamlit app
def main():
    st.title('Candidate Elimination Algorithm')

    # Sidebar - Upload CSV file
    st.sidebar.title('Upload CSV File')
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file is not None:
        # Load data
        data = load_data(uploaded_file)

        # Display data
        st.subheader('Training Data')
        st.write(data)

        # Extract features and target
        # Extract features and target
        target_column = 'enjoySport'  # Replace 'your_target_column_name' with the actual name of your target column
        features = data.drop(columns=[target_column])
        target = data[target_column]  # Use the correct target column

        # Run Candidate Elimination algorithm
        st.subheader('Running Candidate Elimination Algorithm...')
        specific_h, general_h = learn(features.values.tolist(), target.values.tolist())

        # Display final hypotheses
        st.subheader('Final Specialization Hypothesis (s_final)')
        st.write(pd.DataFrame(specific_h))

        st.subheader('Final Generalization Hypothesis (g_final)')
        st.write(pd.DataFrame(general_h))

if __name__ == '__main__':
    main()
