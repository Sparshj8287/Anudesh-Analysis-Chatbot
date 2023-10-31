import streamlit as st
import pandas as pd
import numpy as np
from streamlit_option_menu import option_menu
from menu_styles import HEADER_STYLES
from sqlalchemy import select,join,func,and_,or_,not_,desc,asc,distinct,false,true,text,bindparam,bitwise_not
from sqlalchemy import (
    create_engine,
    MetaData,
    Table,
    Column,
    Integer,
    String,
    PrimaryKeyConstraint,
)
import base64
from pathlib import Path
import plotly.graph_objects as go
import ast
import re
import io
import sys
import openai
import traceback
from collections import Counter

PAGE_TITLE: str = "Anudesh AI4Bharat"
PAGE_ICON: str = "🤖"




st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="wide")


table_schema="""
from sqlalchemy import (
    create_engine,
    MetaData,
    Table,
    Column,
    Integer,
    String,
    PrimaryKeyConstraint,
)

DATABASE_URL = "DATABASE_URL"

engine = create_engine(DATABASE_URL, echo=False)

metadata = MetaData()

Instruction = Table(
    "Instruction",
    metadata,
    Column("instructionID", Integer, primary_key=True),
    Column("intent", String),
    Column("domain", String),
    Column("structure", Integer),
    Column("language", Integer),
    Column("instruction", String),
    Column("examples", String),
)

Interaction = Table(
    "Interaction",
    metadata,
    Column("interactionID", Integer, primary_key=True, autoincrement=True),
    Column("userID", String),
    Column("instructionID", Integer),
    Column("interaction", String),
    Column("model", String),
)

Failed_Interaction = Table(
    "Failed_Interaction",
    metadata,
    Column("interactionID", Integer, primary_key=True, autoincrement=True),
    Column("userID", Integer),
    Column("instructionID", Integer),
    Column("taskID", Integer),
    Column("interaction", String),
    Column("model", String),
)

Failed_Prompts = Table(
    "Failed_Prompts",
    metadata,
    Column("promptID", Integer, primary_key=True, autoincrement=True),
    Column("instructionID", Integer),
    Column("prompt", String),
    Column("reason", Integer),
)

User_Information = Table(
    "User_Info",
    metadata,
    Column("userID", String, primary_key=True),
    Column("name", String),
    Column("display_name", String),
    Column("email", String),
    Column("age", Integer),
    Column("qualification", String),
    Column("local_lang", String),
    Column("city", String),
    Column("state", String),
)

Tasks = Table(
    "TasksDB",
    metadata,
    Column("taskID", Integer),
    Column("jobID", Integer),
    Column("userID", String),
    Column("interactionID", Integer),
    PrimaryKeyConstraint("taskID", "jobID", "userID"),
)

Jobs = Table(
    "Jobs",
    metadata,
    Column("jobsID", Integer, primary_key=True),
    Column("video_details", String),
    Column("courseLink", String),
    Column("courseID", Integer)
)

Discounts = Table(
    "DiscountsDB",
    metadata,
    Column("jobID", Integer),
    Column("discountCode", String),
    Column("used", Integer)
)

User_Discounts = Table(
    "UserDiscountsDB",
    metadata,
    Column("userID", String),
    Column("jobID", Integer),
    Column("discountCode", String),
    PrimaryKeyConstraint("userID", "jobID")
)

User_thinkific = Table(
    "UserThinkificDB",
    metadata,
    Column("userID", String, primary_key=True),
    Column('thinkificID', String)
)
"""


sys_prompt=f"""
You are proficient in SQL, SQLAlchemy, and Python. Based on the given SQL database schema, your goal is to craft functionalities interfacing with this database to deliver accurate outputs based on user queries.

Refer to the table definitions provided in the code below:
{table_schema}

Database Rules:

1. Construct SQL queries using SQLAlchemy and the `select` function.
2. A user must finish 5 tasks to complete a job, indicating a job is finished when taskID reaches 5.
3. Users have a minimum of 1 and a maximum of 3 interactions. 
4. A single interaction is represented as ['random text'] and three interactions appear as [['random text'], ['random text'], ['random text']].

General Guidelines:

1. On initial interaction, provide users with list of sample questions related to the database schema to guide them.
2. For questions unrelated to the database schema, offer an overview of the database's schema structure and provide examples of relevant queries.
3. Notify users if their questions relates to missing data.
4. If a question lacks details, seek further information.
5. Always wrap code outputs within a code block.
6. Keep all code within a single code block cell.
7. STRICTLY FOLLOW THAT YOU HAVE ACCESS ONLY TO WRITE THE QUERIES WHICH ARE USED TO READ THE DATABASE. Never write queries in the function that write to, modify, or delete data in the database.

For Example:
User Query: "How many users are there for a particular job?"

Expected Query:
def total_users_for_job(jobID):
    query = (
        select(func.count(Tasks.c.userID.distinct()))
        .where(Tasks.c.jobID == jobID)
    )
    
    # Execute the query
    with engine.connect() as connection:
        result = connection.execute(query)
        result = result.scalar()
        
    return result

# Example usage:
jobID_to_check = 1
results = total_users_for_job(jobID_to_check)
print(f"Counts of users for each task in job {{jobID_to_check}}: {{results}}")
"""



def get_database():
        DATABASE_URL = "DATABASE_URL"

        engine = create_engine(DATABASE_URL, echo=False)

        metadata = MetaData()

        Instruction = Table(
            "Instruction",
            metadata,
            Column("instructionID", Integer, primary_key=True),
            Column("intent", String),
            Column("domain", String),
            Column("structure", Integer),
            Column("language", Integer),
            Column("instruction", String),
            Column("examples", String),
        )

        Interaction = Table(
            "Interaction",
            metadata,
            Column("interactionID", Integer, primary_key=True, autoincrement=True),
            Column("userID", String),
            Column("instructionID", Integer),
            Column("interaction", String),
            Column("model", String),
        )

        Failed_Interaction = Table(
            "Failed_Interaction",
            metadata,
            Column("interactionID", Integer, primary_key=True, autoincrement=True),
            Column("userID", Integer),
            Column("instructionID", Integer),
            Column("taskID", Integer),
            Column("interaction", String),
            Column("model", String),
        )

        Failed_Prompts = Table(
            "Failed_Prompts",
            metadata,
            Column("promptID", Integer, primary_key=True, autoincrement=True),
            Column("instructionID", Integer),
            Column("prompt", String),
            Column("reason", Integer),
        )

        User_Information = Table(
            "User_Info",
            metadata,
            Column("userID", String, primary_key=True),
            Column("name", String),
            Column("display_name", String),
            Column("email", String),
            Column("age", Integer),
            Column("qualification", String),
            Column("local_lang", String),
            Column("city", String),
            Column("state", String),
        )

        Tasks = Table(
            "TasksDB",
            metadata,
            Column("taskID", Integer),
            Column("jobID", Integer),
            Column("userID", String),
            Column("interactionID", Integer),
            PrimaryKeyConstraint("taskID", "jobID", "userID"),
        )

        Jobs = Table(
            "Jobs",
            metadata,
            Column("jobsID", Integer, primary_key=True),
            Column("video_details", String),
            Column("courseLink", String),
            Column("courseID", Integer)
        )

        Discounts = Table(
            "DiscountsDB",
            metadata,
            Column("jobID", Integer),
            Column("discountCode", String),
            Column("used", Integer)
        )

        User_Discounts = Table(
            "UserDiscountsDB",
            metadata,
            Column("userID", String),
            Column("jobID", Integer),
            Column("discountCode", String),
            PrimaryKeyConstraint("userID", "jobID")
        )

        User_thinkific = Table(
            "UserThinkificDB",
            metadata,
            Column("userID", String, primary_key=True),
            Column('thinkificID', String)
        )
        return engine, metadata, Instruction, Interaction, Failed_Interaction, Failed_Prompts, User_Information, Tasks, Jobs, Discounts, User_Discounts, User_thinkific



engine, metadata, Instruction, Interaction, Failed_Interaction, Failed_Prompts, User_Information, Tasks, Jobs, Discounts, User_Discounts, User_thinkific = get_database()




@st.cache_data(show_spinner=False)
def split_frame(input_df, rows):
    df = [input_df.loc[i : i + rows - 1, :] for i in range(0, len(input_df), rows)]
    return df




@st.cache_data(show_spinner=False)
def get_formatting(df):        
    for i in range(len(df)):
            # print(i)
            user=[]
            bot=[]
        
            # print(eval( df['interaction'][i]))

            try:

                    for j in eval(df['Interaction'][i]):
                            # print(j)
                            user_response, bot_response = j
                            user.append(user_response)
                            bot.append(bot_response)
                    #     print(len(user))
                    if len(user)== 1:
                            final_string=f"""<b>User</b>: {user[0]}<br><b>Bot</b>: {bot[0]}"""
                    elif len(user)== 2:
                            final_string=f"""<b>User</b>: {user[0]}<br><b>Bot</b>: {bot[0]}<br><br><b>User</b>: {user[1]}<br><b>Bot</b>: {bot[1]}"""
                    elif len(user)== 3:
                            final_string=f"""<b>User</b>: {user[0]}<br><b>Bot</b>: {bot[0]}<br><br><b>User</b>: {user[1]}<br><b>Bot</b>: {bot[1]}<br><br><b>User</b>: {user[2]}<br><b>Bot</b>: {bot[2]}"""

                    df['Interaction'][i]=final_string

                    

            except Exception as e:
                    print(e)
                    pass
    return df








@st.cache_data(show_spinner=False)
def get_dataframe():
    try:

        conn = engine.connect()


        stmt = select(
            Interaction.c.userID,
            Instruction.c.instruction,
            Interaction.c.interaction
        ).select_from(
            join(Instruction, Interaction, Instruction.c.instructionID == Interaction.c.instructionID)
        )
    
        results = conn.execute(stmt) 
        
        results_list=results.fetchall()

        df = pd.DataFrame(results_list, columns=['userID', 'Instruction', 'Interaction'])

        final_df=get_formatting(df)

        conn.close()
        return final_df
    except Exception as e:
        print(e)
        return 505
    finally:
        conn.close()


def get_dataframe_from_column(column_name, value):
    stmt = (
        select(
            Interaction.c.userID,
            Instruction.c.instruction,
            Interaction.c.interaction
        )
        .select_from(
            join(Instruction, Interaction, Instruction.c.instructionID == Interaction.c.instructionID)
        )
        .where(
            getattr(Instruction.c, column_name) == value
        )
    )
    with engine.connect() as conn:
        result = conn.execute(stmt).fetchall()
    df = pd.DataFrame(result, columns=['userID', 'Instruction', 'Interaction'])
    final_df = get_formatting(df)
    return final_df


@st.cache_data(show_spinner=False)
def get_interactions_for_job(job_id):

    with engine.connect() as connection:
        task_query = select(Tasks.c.userID).where(Tasks.c.jobID == job_id)
        user_ids = connection.execute(task_query).fetchall()


        distinct_user_ids = list(set([row[0] for row in user_ids]))

        if distinct_user_ids:


            interaction_query = (
        select(
            Interaction.c.userID,
            Instruction.c.instruction,
            Interaction.c.interaction
        )
        .select_from(
            join(Instruction, Interaction, Instruction.c.instructionID == Interaction.c.instructionID)
        ).where(Interaction.c.userID.in_(distinct_user_ids))
    )
            # interaction_query = select(Interaction).where(Interaction.c.userID.in_(distinct_user_ids))
            interactions = connection.execute(interaction_query).fetchall()

            df = pd.DataFrame(interactions, columns=['userID', 'Instruction', 'Interaction'])
            final_df = get_formatting(df)
            return final_df
        else:
            print("No interactions found for the given jobID!")
            return []



@st.cache_data(show_spinner=False)
def get_user_interactions(user_name):
    """
    Retrieve all interactions done by the user based on the given user name.

    :param user_name: The name of the user
    :return: List of interactions for the user
    """

    # Create a connection to the engine
    with engine.connect() as connection:
        # Use the User_Information table to get the userID based on the user name
        user_query = select(User_Information.c.userID).where(User_Information.c.name == user_name)
        user_result = connection.execute(user_query).fetchone()

        if user_result:
            user_id = user_result[0]

            # Use the userID to retrieve all interactions from the Interaction table
            interaction_query = select(
            Interaction.c.userID,
            Instruction.c.instruction,
            Interaction.c.interaction
        ).select_from(
            join(Instruction, Interaction, Instruction.c.instructionID == Interaction.c.instructionID)
        ).where(Interaction.c.userID == user_id)


            interactions = connection.execute(interaction_query).fetchall()
            # print(interactions)

            # Convert the interactions to a list of dictionaries for easy readability
            df = pd.DataFrame(interactions,columns=['userID','Instruction','Interaction'])

            final_df=get_formatting(df)

            return final_df,interactions
        else:
            print("User not found!")
            return []






@st.cache_data(show_spinner=False)
def dataframe_from_intent(intent):
    return get_dataframe_from_column('intent', intent)




@st.cache_data(show_spinner=False)
def dataframe_from_domain(domain):
    return get_dataframe_from_column('domain', domain)


@st.cache_data(show_spinner=False)
def dataframe_from_language(language):
    return get_dataframe_from_column('language', language)



def pagination(df):
        dataset = df
        # st.markdown(df.to_html(escape=False), unsafe_allow_html=True)


        top_menu = st.columns(6)
        with top_menu[0]:
            sort = st.radio("Sort Data", options=["Yes", "No"], horizontal=1, index=1)
        if sort == "Yes":
            with top_menu[1]:
                sort_field = st.selectbox("Sort By", options=dataset.columns)
            with top_menu[2]:
                sort_direction = st.radio(
                "Direction", options=["⬆️", "⬇️"], horizontal=True
            )
            dataset = dataset.sort_values(
            by=sort_field, ascending=sort_direction == "⬆️", ignore_index=True
        )
            
        with top_menu[5]:
            batch_size = st.selectbox("Page Size", options=[15, 25, 50, 100])

        with top_menu[4]:
            total_pages = (
            int(len(dataset) / batch_size) if int(len(dataset) / batch_size) > 0 else 1
        )
            current_page = st.number_input(
            "Page", min_value=1, max_value=total_pages, step=1
        )
        
        with top_menu[3]:
            st.markdown(f"Page **{current_page}** of **{total_pages}** ")   
        try:
            pagination = st.container()

        #     bottom_menu = st.columns((4, 1, 1))
        #     with bottom_menu[2]:
        #         batch_size = st.selectbox("Page Size", options=[15, 25, 50, 100])
        #     with bottom_menu[1]:
        #         total_pages = (
        #         int(len(dataset) / batch_size) if int(len(dataset) / batch_size) > 0 else 1
        #     )
        #         current_page = st.number_input(
        #         "Page", min_value=1, max_value=total_pages, step=1
        #     )
        # with bottom_menu[0]:
        #     st.markdown(f"Page <b>{current_page}</b> of <b>{total_pages}</b> ")



            pages = split_frame(dataset, batch_size)
            new_dataframe=pages[current_page - 1]
            data=st.markdown(new_dataframe.to_html(escape=False), unsafe_allow_html=True)
            pagination.dataframe(data, use_container_width=True)

        except Exception as e:
            print(e)
            pass
        # pagination.dataframe(data=pages[current_page - 1], use_container_width=True)
        # st.markdown(df.to_html(escape=False), unsafe_allow_html=True)
        # pagination.table(data=data)
        # st.markdown(pagination.table(data=pages[current_page - 1]),unsafe_allow_html=True)


def remove_duplicates(input_list):
    seen = set()
    output_list = []
    for item in input_list:
        if item not in seen:
            output_list.append(item)
            seen.add(item)
    return output_list


def img_to_bytes(img_path):
        img_bytes = Path(img_path).read_bytes()
        encoded = base64.b64encode(img_bytes).decode()
        return encoded



def img_to_html(img_path):
        img_html = """
        <div style="display: flex; justify-content: center;">
            <img src='data:image/png;base64,{}' class='img-fluid' alt='centered image'>
        </div>
        """.format(img_to_bytes(img_path))
        return img_html



@st.cache_data(show_spinner=False)
def total_distinct_users():
    query = select(func.count(User_Information.c.userID.distinct()))
    
    # Execute the query
    with engine.connect() as connection:
        result = connection.execute(query)
        result = result.scalar()
        
    return result



@st.cache_data(show_spinner=False)
def finished_users_count(jobID):
    query = (
        select(func.count(Tasks.c.userID.distinct()))
        .where(
            (Tasks.c.jobID == jobID)
            & (Tasks.c.taskID == 5)  
        )
    )

    with engine.connect() as connection:
        result = connection.execute(query)
        result = result.scalar()


    return result


@st.cache_data(show_spinner=False)
def average_num_of_tasks_com(jobID):

    query = (
            select(Tasks.c.taskID)
            .where(
                (Tasks.c.jobID == jobID)
            )
        )
        
    with engine.connect() as connection:
            result = connection.execute(query)
            results = result.fetchall()
            results = [i[0] for i in results]

    results_sum= sum(results)

    results_len= len(results)
    avg= results_sum/results_len

    return avg



@st.cache_data(show_spinner=False)
def average_number_of_ins_per_job(jobID):
    query = (
        select(Interaction.c.interaction)
        .join(Tasks, Tasks.c.interactionID == Interaction.c.interactionID)
        .where(Tasks.c.jobID == jobID)
    )
    
    with engine.connect() as connection:
        interactions = [row[0] for row in connection.execute(query).fetchall()]

    lengths = [len(ast.literal_eval(interaction)) for interaction in interactions]

    count = Counter(lengths)

    number = sum([count[i] * i for i in [1, 2, 3]])

    return number / len(lengths)


@st.cache_data(show_spinner=False)
def average_number_of_ins_per_tasks(taskID):
    query = (
        select(Interaction.c.interaction)
        .join(Tasks, Tasks.c.interactionID == Interaction.c.interactionID)
        .where(Tasks.c.taskID == taskID)
    )
    
    with engine.connect() as connection:
        interactions = [row[0] for row in connection.execute(query).fetchall()]

    lengths = [len(ast.literal_eval(interaction)) for interaction in interactions]

    count = Counter(lengths)

    number = sum([count[i] * i for i in [1, 2, 3]])

    return number / len(lengths)




def extract_code(text):
    code_blocks = re.findall(r'```(.*?)```', text, re.DOTALL)
    code_blocks = [code.replace('python', '').strip() for code in code_blocks]
    return code_blocks

def gpt_completion(prompt):
    all_messages=openai.ChatCompletion.create(
    engine=st.session_state["openai_model"],
    messages=[
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages
            ]
        )
    return all_messages["choices"][0]["message"]["content"]


def execute_gpt_generated_code(generated_code,question):
    count=0
    code_out = io.StringIO() 
    sys.stdout = code_out  
    try:
        exec(generated_code)
        count=1
    except Exception as e:
        code_out.write(f"""
You are proficient in SQL, SQLAlchemy, and Python. Based on the given SQL database schema, your goal is to craft functionalities interfacing with this database to deliver accurate outputs based on user queries.

Refer to the table definitions provided in the code below:  
{table_schema}

The user asked the following question:
{question}

You generated this python code:
{generated_code}

It fails with the following error:\n
""")
        code_out.write(traceback.format_exc())

        code_out.write("\nCorrect the python code and return a new python code that fixes the above mentioned error. Do not generate the same code again.")
        count=0

    finally:
        sys.stdout = sys.__stdout__  
    return count,code_out.getvalue()  


def get_gpt_response(prompt):
    response = gpt_completion(prompt)
    print(response)
    return response


def execute_and_check(code, prompt):
    print("\n\n*************************** Length of Code ***************************\n\n")
    print(len(code))
    number, output = execute_gpt_generated_code(code[0], prompt)
    return number, output


def handle_error(final_output):
    error_sol = gpt_completion(final_output)
    return extract_code(error_sol)


def main_loop(prompt):
    full_response = get_gpt_response(prompt)
    code = extract_code(full_response)
    count=0

    if not code:
        return -1, full_response

    for _ in range(10):  # Try at most 3 times

        number, final_output = execute_and_check([c for c in code], prompt)

        
        count+=1

        print(f"\n\n*************************** Attempt {count} ***************************\n\n")

        print(final_output)

        if number == 1:
            return 1, final_output
        
        code = handle_error(final_output)

        if not code:
            break

    return 0, "Ask the query again."




def main():
    st.markdown(img_to_html('temp_resized_image.png'), unsafe_allow_html=True)

    st.markdown(f"<h1 style='text-align: center; color: #EE6633;'>ANUDESH ANALYSIS</h1>", unsafe_allow_html=True)

    page = st.sidebar.selectbox('Select page',['Quality Analysis','Quantative Analysis','Anudesh Analysis Chatbot']) 

    if page == 'Quality Analysis':
                selected_footer = option_menu(
                    menu_title=None,
                    options=[
                        'All',
                        'Users',
                        'Intent',
                        'Domain',
                        'Lang',
                        'Job'
                    ],
                    icons=["info-circle", "list-task","chat-square-text","chat-square-text","chat-square-text","chat-square-text"],  # https://icons.getbootstrap.com/
                    menu_icon="cast",
                    default_index=0,
                    orientation="horizontal",
                    styles=HEADER_STYLES
                )



                if selected_footer=='All':
                    # with st.container():
                    df=get_dataframe()
                    pagination(df)
                        # st.dataframe(df)

                elif selected_footer=="Intent":
                    final_intents=[]
                    count=0

                    with open('intent_list.txt') as f:
                        intent_list = f.read().splitlines()
                        intent_list=remove_duplicates(intent_list)
                        for i in intent_list:
                            string= i.split('(')[0]
                            final_intents.append(string)

                    final_intents.insert(0,'Select')
                    choice = st.selectbox("Choose an option:", final_intents, index=0)
                    if choice == 'Select':
                        st.warning("Please select an option.")
                    elif choice != 'Select':
                        st.success(f"You have successfully selected {choice}!")
                        with st.spinner():
                            for index,i in enumerate(final_intents):

                                if choice==i:
                                    count=index
                                    print(count)
                                    break

                            intent=intent_list[count-1]
                            df=dataframe_from_intent(intent)
                            pagination(df)


                elif selected_footer=="Domain":

                    with open('domains.txt','r') as f:
                        domains = f.read().splitlines()
                        domains=remove_duplicates(domains)
                        domains.insert(0,'Select')
                        choice = st.selectbox("Choose an option:", domains, index=0)
                        if choice == 'Select':
                            st.warning("Please select an option.")
                        elif choice != 'Select':
                            st.success(f"You have successfully selected {choice}!")
                            with st.spinner():
                                df=dataframe_from_domain(choice)
                                pagination(df)

                elif selected_footer=="Lang":
                    
                    with open('language.txt','r') as f:
                        languages = f.read().splitlines()
                        languages=remove_duplicates(languages)
                        languages.insert(0,'Select')
                        choice = st.selectbox("Choose an option:", languages, index=0)
                        if choice == 'Select':
                            st.warning("Please select an option.")
                        elif choice != 'Select':
                            st.success(f"You have successfully selected {choice}!")
                            with st.spinner():
                                df=dataframe_from_language(choice)
                                pagination(df)

                elif selected_footer=="Users":
                    with open('user_name.txt','r') as f:
                        users = f.read().splitlines()
                        users=remove_duplicates(users)
                        users.insert(0,'Select')
                        choice = st.selectbox("Choose an option:", users, index=0)
                        if choice == 'Select':
                            st.warning("Please select an option.")
                        elif choice != 'Select':
                            st.success(f"You have successfully selected {choice}!")
                            with st.spinner():
                                df,interactions=get_user_interactions(choice)
                                if interactions==[]:
                                    st.error("No data found for the selected user.")
                                if interactions!=[]:
                                    pagination(df)

                elif selected_footer=="Job":
                        jobs=[1,2,3,4]
                        jobs.insert(0,'Select')
                        choice = st.selectbox("Choose an option:", jobs, index=0)
                        if choice == 'Select':
                            st.warning("Please select an option.")
                        elif choice != 'Select':
                            st.success(f"You have successfully selected {choice}!")
                            with st.spinner():
                                print(choice)
                                df=get_interactions_for_job(choice)
                                pagination(df)


    elif page == 'Quantative Analysis':
        total_users=total_distinct_users()
        st.markdown(f"<h5 style='text-align: center; color: black;'>Total numbers of Users: {total_users}</h5>", unsafe_allow_html=True)

        with st.container():
            col1, col2 = st.columns([5,5])
            with col1:
                with st.spinner():
                    finished_users= []
                    for i in range(1,5):
                        finished_users.append(finished_users_count(i))
                    
                    categories = ["Job 1", "Job 2", "Job 3", "Job 4"]
                    values = finished_users

                    colors = ["#FF8E66", "#FFA580", "#FFC4A6","#FFD8C1"]

                    fig = go.Figure(data=[
                        go.Bar(x=categories, y=values, marker_color=colors, text=values, textposition='auto')
                    ])

                    fig.update_layout(
                        title_text="Number of users who successfully completed a Job",
                        font_family="monospace",
                        font_color="#3B3B3B",
                        plot_bgcolor="#FEF8F4",
                        paper_bgcolor="#FEF8F4",
                        title_font_color="#3B3B3B",
                        xaxis_title="Categories",
                        yaxis_title="Number of Users",
                        xaxis = dict(tickvals = [0,1,2,3], ticktext = categories, tickcolor="#3B3B3B"),
                        yaxis = dict(tickcolor="#3B3B3B"),
                    )

                    # # Display on Streamlit
                    # col1, col2, col3 = st.columns([1,6,1])
                    # with col2:
                    st.plotly_chart(fig, use_container_width=True)

                    average_num_ins_per_job=[]
                    for i in range(1,5):
                        average_num_ins_per_job.append(average_number_of_ins_per_job(i))

                    categories = ["Job 1", "Job 2", "Job 3", "Job 4"]
                    values = average_num_ins_per_job

                    colors = ["#FF8E66", "#FFA580", "#FFC4A6","#FFD8C1"]

                    fig = go.Figure(data=[
                        go.Bar(x=categories, y=values, marker_color=colors, text=values, textposition='auto')
                    ])

                    fig.update_layout(
                        title_text="Average number of Interactions per Job",
                        font_family="monospace",
                        font_color="#3B3B3B",
                        plot_bgcolor="#FEF8F4",
                        paper_bgcolor="#FEF8F4",
                        title_font_color="#3B3B3B",
                        xaxis_title="Categories",
                        yaxis_title="Number of Interactions",
                        xaxis = dict(tickvals = [0,1,2,3], ticktext = categories, tickcolor="#3B3B3B"),
                        yaxis = dict(tickcolor="#3B3B3B"),
                    )

                    # # Display on Streamlit
                    # col1, col2, col3 = st.columns([1,6,1])
                    # with col2:
                    st.plotly_chart(fig, use_container_width=True)

            with col2:
                with st.spinner():
                    average_num_of_tasks_completed=[]
                    for i in range(1,5):
                        average_num_of_tasks_completed.append(average_num_of_tasks_com(i))

                    categories = ["Job 1", "Job 2", "Job 3", "Job 4"]
                    values = average_num_of_tasks_completed

                    colors = ["#FF8E66", "#FFA580", "#FFC4A6","#FFD8C1"]

                    fig = go.Figure(data=[
                        go.Bar(x=categories, y=values, marker_color=colors, text=values, textposition='auto')
                    ])

                    fig.update_layout(
                        title_text="Average number of Tasks completed per Job",
                        font_family="monospace",
                        font_color="#3B3B3B",
                        plot_bgcolor="#FEF8F4",
                        paper_bgcolor="#FEF8F4",
                        title_font_color="#3B3B3B",
                        xaxis_title="Categories",
                        yaxis_title="Number of Tasks",
                        xaxis = dict(tickvals = [0,1,2,3], ticktext = categories, tickcolor="#3B3B3B"),
                        yaxis = dict(tickcolor="#3B3B3B"),
                    )

                    # # Display on Streamlit

                    st.plotly_chart(fig, use_container_width=True)

                    average_num_ins_per_tasks=[]
                    for i in range(1,6):
                        average_num_ins_per_tasks.append(average_number_of_ins_per_tasks(i))

                    categories = ["Task 1", "Task 2", "Task 3", "Task 4", "Task 5"]
                    values = average_num_ins_per_tasks

                    colors = ["#FF8E66", "#FFA580", "#FFC4A6","#FFD8C1","#FFE9D9"]

                    fig = go.Figure(data=[
                        go.Bar(x=categories, y=values, marker_color=colors, text=values, textposition='auto')
                    ])

                    fig.update_layout(
                        title_text="Average number of Interactions per Task",
                        font_family="monospace",
                        font_color="#3B3B3B",
                        plot_bgcolor="#FEF8F4",
                        paper_bgcolor="#FEF8F4",
                        title_font_color="#3B3B3B",
                        xaxis_title="Categories",
                        yaxis_title="Number of Interactions",
                        xaxis = dict(tickvals = [0,1,2,3,4], ticktext = categories, tickcolor="#3B3B3B"),
                        yaxis = dict(tickcolor="#3B3B3B"),
                    )

                    # # Display on Streamlit
                    # col1, col2, col3 = st.columns([1,6,1])
                    # with col2:
                    st.plotly_chart(fig, use_container_width=True)


    elif page == 'Anudesh Analysis Chatbot':
        correct_username = st.secrets["users"]["username"]
        correct_password = st.secrets["users"]["password"]

        logged_in = st.session_state.get('logged_in', False)

        if not logged_in:
            with st.form(key="login_form"):
                # Input fields for user input
                username = st.text_input("Enter your username")
                password = st.text_input("Enter your password", type='password')
                submit_button = st.form_submit_button(label="Login")

                if submit_button:
                    if username == correct_username and password == correct_password:
                        logged_in = True
                        st.session_state.logged_in = True  
                    else:
                        st.error("The username or password you entered is incorrect")

        else:
            # st.success("Logged in successfully")
            st.markdown(f"<h5 style='text-align: center; color: #EE6633;'>ANUDESH ANALYSIS CHATBOT</h5>", unsafe_allow_html=True)

            openai.api_type = "azure"
            openai.api_base = "OPENAI_API_BASE"
            openai.api_version = "OPENAI_API_VERSION"
            openai.api_key = st.secrets["OPENAI_API_KEY"]

            if "openai_model" not in st.session_state:
                st.session_state["openai_model"] = "prompt-chat-gpt4"

            if "messages" not in st.session_state:
                st.session_state.messages = []

            for message in st.session_state.messages:
                if message["role"] != "system":
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])

            st.session_state.messages.append({"role": "system", "content": sys_prompt})

            if prompt := st.chat_input("Enter your Question here"):
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        message_placeholder = st.empty()
                        number, response = main_loop(prompt)

                        print("\n\n********************* RESPONSE ****************************\n\n")
                        print(response)

                        print("\n\n********************* NUMBER ******************************\n\n")
                        print(number)
                        message_placeholder.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})



if __name__ == "__main__":
    main()