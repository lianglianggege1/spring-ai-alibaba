package com.alibaba.cloud.ai.graph.agent.model;


import com.alibaba.cloud.ai.graph.OverAllState;
import com.alibaba.cloud.ai.graph.agent.ReactAgent;
import com.alibaba.cloud.ai.graph.checkpoint.savers.MemorySaver;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfEnvironmentVariable;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.minimax.MiniMaxChatModel;
import org.springframework.ai.minimax.MiniMaxChatOptions;
import org.springframework.ai.minimax.api.MiniMaxApi;

import java.util.List;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assertions.fail;

@EnabledIfEnvironmentVariable(named = "AI_MINIMAX_API_KEY", matches = ".+")
@Slf4j
public class ReactAgentMiniMaxTest {

    private ChatModel chatModel;

    @Data
    public static class PoemOutput {

        private String title;

        private String content;

        private String style;

    }


    @BeforeEach
    void setUp() {
        MiniMaxApi miniMaxApi = new MiniMaxApi(
                System.getenv("AI_MINIMAX_API_KEY")
        );


        this.chatModel = new MiniMaxChatModel(miniMaxApi,
                MiniMaxChatOptions.builder()
                        .model("MiniMax-M2.5")
                        .build()
        );
    }

    @Test
    public void testReactAgent() throws Exception {

        ReactAgent agent = ReactAgent.builder().name("single_agent").model(chatModel).saver(new MemorySaver()).build();

        try {
            Optional<OverAllState> result = agent.invoke("帮我写一篇100字左右散文。");
            Optional<OverAllState> result2 = agent.invoke(new UserMessage("帮我写一首现代诗歌。"));
            Optional<OverAllState> result3 = agent.invoke("帮我写一首现代诗歌2。");

            assertTrue(result.isPresent(), "First result should be present");
            OverAllState state1 = result.get();
            assertTrue(state1.value("messages").isPresent(), "Messages should be present in first result");
            assertEquals(2, ((List)state1.value("messages").get()).size(), "There should be 2 messages in the first result");
            Object messages1 = state1.value("messages").get();
            assertNotNull(messages1, "Messages should not be null in first result");

            assertTrue(result2.isPresent(), "Second result should be present");
            OverAllState state2 = result2.get();
            assertTrue(state2.value("messages").isPresent(), "Messages should be present in second result");
            assertEquals(4, ((List<?>)state2.value("messages").get()).size(), "There should be 2 messages in the first result");
            Object messages2 = state2.value("messages").get();
            assertNotNull(messages2, "Messages should not be null in second result");

            assertNotEquals(messages1, messages2, "Results should be different for different inputs");

            System.out.println(result.get());
            System.out.println(result2.get());
            System.out.println(result2.get());

        }
        catch (java.util.concurrent.CompletionException e) {
            e.printStackTrace();
            fail("ReactAgent execution failed: " + e.getMessage());
        }
    }

}
