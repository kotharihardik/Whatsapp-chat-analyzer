import streamlit as st
import preprocessor, helper
import matplotlib.pyplot as plt
import seaborn as sns

from helper import get_response_from_llm
from preprocessor import npreprocess

# Set up the Streamlit app
st.sidebar.title("Whatsapp Chat Analyzer")

# File uploader for the chat file
uploaded_file = st.sidebar.file_uploader("Choose a file")

if uploaded_file is not None:
    # Reading the uploaded file
    bytes_data = uploaded_file.getvalue()
    data = bytes_data.decode("utf-8")
    
    # Preprocessing the data using the preprocessor
    try:
        df = preprocessor.preprocess(data)
        st.write("✅ Preprocessed chat data:")
        st.dataframe(df.head())  # Replaced print with st.dataframe

    except Exception as e:
        st.error(f"Error in preprocessing the data: {e}")
        st.stop()

    # Fetching the list of users from the dataframe
    user_list = df['user'].unique().tolist()
    if 'group_notification' in user_list:
        user_list.remove('group_notification')

    user_list.sort()
    user_list.insert(0, "Overall")  # Adding 'Overall' for group-level analysis

    selected_user = st.sidebar.selectbox("Show analysis wrt", user_list)

    # Show Analysis Button with session state
    if "show_analysis" not in st.session_state:
        st.session_state.show_analysis = False

    if st.sidebar.button("Show Analysis"):
        st.session_state.show_analysis = True

    if st.session_state.show_analysis:
        try:
            num_messages, words, num_media_messages, num_links = helper.fetch_stats(selected_user, df)

            st.title("📊 Top Statistics")
            col1, col2, col3 = st.columns(3)

            col1.metric("Total Messages", num_messages)
            col2.metric("Total Words", words)

            # Only show media if count > 0
            # if num_media_messages > 0:
            #     col3.metric("Media Shared", num_media_messages)
            # else:
            #     col3.metric("Media Shared", "—")

            col3.metric("Links Shared", num_links)


            # Monthly timeline
            st.title("Monthly Timeline")
            timeline = helper.monthly_timeline(selected_user, df)
            fig, ax = plt.subplots()
            ax.plot(timeline['time'], timeline['message'], color='green')
            plt.xticks(rotation='vertical')
            fig.tight_layout()
            st.pyplot(fig)

            # Daily timeline
            st.title("Daily Timeline")
            daily_timeline = helper.daily_timeline(selected_user, df)
            fig, ax = plt.subplots()
            ax.plot(daily_timeline['only_date'], daily_timeline['message'], color='black')
            plt.xticks(rotation='vertical')
            fig.tight_layout()
            st.pyplot(fig)

            # Activity map
            st.title('Activity Map')
            col1, col2 = st.columns(2)

            with col1:
                st.header("Most busy day")
                busy_day = helper.week_activity_map(selected_user, df)
                fig, ax = plt.subplots()
                ax.bar(busy_day.index, busy_day.values, color='purple')
                plt.xticks(rotation='vertical')
                fig.tight_layout()
                st.pyplot(fig)

            with col2:
                st.header("Most busy month")
                busy_month = helper.month_activity_map(selected_user, df)
                fig, ax = plt.subplots()
                ax.bar(busy_month.index, busy_month.values, color='orange')
                plt.xticks(rotation='vertical')
                fig.tight_layout()
                st.pyplot(fig)

            # Weekly activity heatmap
            st.title("Weekly Activity Map")
            user_heatmap = helper.activity_heatmap(selected_user, df)
            if not user_heatmap.empty:
                fig, ax = plt.subplots()
                sns.heatmap(user_heatmap, ax=ax)
                fig.tight_layout()
                st.pyplot(fig)
            else:
                st.warning("No activity data available to generate heatmap.")

            # Most busy users (group level)
            if selected_user == 'Overall':
                st.title('Most Busy Users')
                x, new_df = helper.most_busy_users(df)
                col1, col2 = st.columns(2)

                with col1:
                    fig, ax = plt.subplots()
                    ax.bar(x.index, x.values, color='red')
                    plt.xticks(rotation='vertical')
                    fig.tight_layout()
                    st.pyplot(fig)

                with col2:
                    st.dataframe(new_df)

            # Wordcloud
            st.title("Wordcloud")
            df_wc = helper.create_wordcloud(selected_user, df)
            fig, ax = plt.subplots()
            ax.imshow(df_wc)
            ax.axis("off")
            fig.tight_layout()
            st.pyplot(fig)

            # Most common words
            st.title('Most Common Words')
            most_common_df = helper.most_common_words(selected_user, df)
            fig, ax = plt.subplots()
            ax.barh(most_common_df[0], most_common_df[1])
            plt.xticks(rotation='vertical')
            fig.tight_layout()
            st.pyplot(fig)

            # Emoji analysis
            st.title("Emoji Analysis")
            emoji_df = helper.emoji_helper(selected_user, df)

            col1, col2 = st.columns(2)
            with col1:
                st.dataframe(emoji_df)

            with col2:
                fig, ax = plt.subplots()
                ax.pie(emoji_df[1].head(), labels=emoji_df[0].head(), autopct="%0.2f")
                fig.tight_layout()
                st.pyplot(fig)

            # 💬 LLM-Based Chat Insight Section
            st.title("💬 AI-Powered Chat Insights")

            st.subheader("💡 Ask a Question About This Chat")

            # User input for LLM question
            user_question = st.text_input("Type your question here...", key="safe_input")

            # # Show debug
            # st.write(f"DEBUG: You typed — {user_question!r}")

            if st.button("Ask"):
                if user_question.strip() != "":
                    with st.spinner("Getting answer from AI..."):
                        try:
                            short_chat_excerpt = data[:15000]  # Limit context for LLM
                            full_prompt = f"Here is a WhatsApp chat excerpt:\n\n{short_chat_excerpt}\n\nQuestion: {user_question}\nAnswer briefly and clearly:"
                            answer = get_response_from_llm(full_prompt, max_tokens=350)
                            st.session_state["test_answer"] = answer.strip()
                        except Exception as e:
                            st.error(f"❌ LLM Error: {e}")
                            st.session_state["test_answer"] = None
                else:
                    st.warning("Please enter a question before clicking Ask.")

            # Show the result if available
            if "test_answer" in st.session_state and st.session_state["test_answer"]:
                st.success("🧠 AI Response:")
                st.markdown(st.session_state["test_answer"])

        except Exception as e:
            st.error(f"Error occurred while generating the analysis: {e}")
