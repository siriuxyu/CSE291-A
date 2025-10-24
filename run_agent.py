import asyncio
from agent.graph import graph
from agent.context import Context
from agent.state import InputState
from langchain_core.messages import HumanMessage, AIMessage

async def main_token_stream():
    input = InputState(
        messages=[HumanMessage(content="what is the weather in San Diego?")]
    )

    async for chunk in graph.astream(
        input=input,
        context=Context(system_prompt="You are a helpful AI assistant.")
    ):
        print(chunk)

SESSION_ID = "test_session1"

async def ask(text: str):
    input_state = InputState(messages=[HumanMessage(content=text)])

    async for event in graph.astream(
        input=input_state,
        context=Context(system_prompt="You are a helpful AI assistant."),
        config={"configurable": {"thread_id": SESSION_ID,}},
    ):
        print(event)

    print("======End of Response======")

async def main():
    # await ask("Please read the following text and answer the questions that follow: [Text Start] Is comprehension the same whether a person reads a text onscreen or on paper? ... The reasons relate to... reduced concentration... When reading texts of several hundred words or more, learning is generally more successful when it's on paper... The benefits of print reading particularly shine through when experimenters move from... simple tasks... to ones that require mental abstraction... The differences... relate to paper's physical properties... People often link their memory... to how far into the book... But equally important is the mental aspect... 'shallowing hypothesis.' According to this theory, people approach digital texts with a mindset suited to social media... and devote less mental effort... Audio and video can feel more engaging... psychologists have demonstrated that when adults read news stories, they remember more... Digital texts, audio and video all have educational roles... However, for maximizing learning... educators shouldn't assume all media are the same... [Text End]. Now, Q1: What does the underlined phrase 'shine through' in paragraph 2 mean? (A) Seem unlikely to last. (B) Seem hard to explain. (C) Become ready to use. (D) Become easy to notice.")
    # await ask("Q2: What does the shallowing hypothesis assume? (A) Readers treat digital texts lightly. (B) Digital texts are simpler to understand. (C) People select digital texts randomly. (D) Digital texts are suitable for social media.")
    # await ask("Q3: Why are audio and video increasingly used by university teachers? (A) They can hold students' attention. (B) They are more convenient to prepare. (C) They help develop advanced skills. (D) They are more informative than text.")
    # await ask("Q4: What does the author imply in the last paragraph? (A) Students should apply multiple learning techniques. (B) Teachers should produce their own teaching material. (C) Print texts cannot be entirely replaced in education. (D) Education outside the classroom cannot be ignored.")
    # await ask("I'm going to Hawaii for 5 days. I prefer the sea rather than the mountains and I want to go diving at least twice. I'm also on a medium budget. Could you give me a plan?")
    # await ask("What is the square root of (ln(18) ^ 3) / 3.14?")
    # await ask("How is the weather in San Diego right now?")
    # await ask("Translate 'Have a nice day!' to French.")
    # await ask("Read the news : https://www.yahoo.com/news/article/aws-recovering-after-major-outage-affects-apps-and-websites-including-snapchat-and-ring-133005156.html. Which server region is experiencing an issue?")
    # await ask("Now give me a detailed schedule for Day 2 of the plan.")
    # await ask("What is the weather in Beijing today?")
    # await ask("Thanks. Now, back to the trip. Please compare the water temperature in Hawaii for my trip with the current water temperature in my living place.")
    # await ask("What is the weather in Tokyo today? Please translate the weather report to Japanese.")
    # await ask(r"A portfolio was worth $250,000 last year. If it had an 8.5% return this year, how much did it increase in value?")
    # await ask(r"From that increase, subtract 20% for capital gains tax. How much is left after tax?")
    await ask("I need some help. How do I say, 'I would like to book a double room'?")
    await ask("I am going to Paris for business, how is the weather there?")
    await ask("Thanks. Another one, 'Is breakfast included?'")

if __name__ == "__main__":
    asyncio.run(main())