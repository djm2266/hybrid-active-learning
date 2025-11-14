#!/usr/bin/env python3

from src.utils import load_config, ensure_directories, get_llm_provider


def main():
    config = load_config()
    ensure_directories(config)
    llm = get_llm_provider(config)

    print("\nLLM provider type:", type(llm))

    messages = [
        {
            "role": "system",
            "content": "You are a test model. Reply ONLY with the text: TEST_OK"
        },
        {
            "role": "user",
            "content": "Say TEST_OK"
        }
    ]

    print("\n=== Calling llm.chat_completion(...) ===")
    resp = llm.chat_completion(messages, temperature=0, max_tokens=10)
    print("\n=== Raw response from provider ===")
    print(repr(resp))


if __name__ == "__main__":
    main()
