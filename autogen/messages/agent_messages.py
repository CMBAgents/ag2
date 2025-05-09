# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from abc import ABC
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Callable, Literal, Optional, Union
from uuid import UUID

from pydantic import BaseModel
from termcolor import colored

from IPython.display import display
from IPython.display import Markdown

from ..code_utils import content_str
from ..oai.client import OpenAIWrapper
from .base_message import BaseMessage, wrap_message
from ..cmbagent_utils import cmbagent_debug, cmbagent_color_dict, cmbagent_default_color

if TYPE_CHECKING:
    from ..agentchat.agent import Agent
    from ..coding.base import CodeBlock


__all__ = [
    "ClearAgentsHistoryMessage",
    "ClearConversableAgentHistoryMessage",
    "ConversableAgentUsageSummaryMessage",
    "ConversableAgentUsageSummaryNoCostIncurredMessage",
    "ExecuteCodeBlockMessage",
    "ExecuteFunctionMessage",
    "FunctionCallMessage",
    "FunctionResponseMessage",
    "GenerateCodeExecutionReplyMessage",
    "GroupChatResumeMessage",
    "GroupChatRunChatMessage",
    "PostCarryoverProcessingMessage",
    "SelectSpeakerMessage",
    "SpeakerAttemptFailedMultipleAgentsMessage",
    "SpeakerAttemptFailedNoAgentsMessage",
    "SpeakerAttemptSuccessfulMessage",
    "TerminationAndHumanReplyMessage",
    "TextMessage",
    "ToolCallMessage",
    "ToolResponseMessage",
]

MessageRole = Literal["assistant", "function", "tool"]


class BasePrintReceivedMessage(BaseMessage, ABC):
    content: Union[str, int, float, bool]
    sender_name: str
    recipient_name: str

    def print(self, f: Optional[Callable[..., Any]] = None) -> None:
        f = f or print
        # f(f"{colored(self.sender_name, 'yellow')} (to {self.recipient_name}):\n", flush=True)
        # f(f"{colored(f'Message from {self.sender_name}:\n', 'yellow')}", flush=True)
        color = cmbagent_color_dict.get(self.sender_name, cmbagent_default_color)
        if cmbagent_debug:
            message = f"Message from {self.sender_name}:\n"  # Store in a variable
            f(colored(message, color), flush=True)  # Apply `colored` separately
        else:
            if self.sender_name != "_Swarm_Tool_Executor":
                message = f"Message from {self.sender_name}:\n"  # Store in a variable
                f(colored(message, color), flush=True)  # Apply `colored` separately


@wrap_message
class FunctionResponseMessage(BasePrintReceivedMessage):
    name: Optional[str] = None
    role: MessageRole = "function"
    content: Union[str, int, float, bool]

    def print(self, f: Optional[Callable[..., Any]] = None) -> None:
        f = f or print
        super().print(f)

        id = self.name or "No id found"
        if cmbagent_debug:
            func_print = f"***** Response from calling {self.role} ({id}) *****"
            f(colored(func_print, "blue"), flush=True)
            # f(self.content, flush=True)
            display(Markdown(self.content))
            f(colored("*" * len(func_print), "blue"), flush=True)
        else:
            # f(self.content, flush=True)
            display(Markdown(self.content))

        if cmbagent_debug:
            f("\n", "-" * 80, flush=True, sep="")


class ToolResponse(BaseModel):
    tool_call_id: Optional[str] = None
    role: MessageRole = "tool"
    content: Union[str, int, float, bool]

    def print(self, f: Optional[Callable[..., Any]] = None) -> None:
        f = f or print
        id = self.tool_call_id or "No id found"
        if cmbagent_debug:
            tool_print = f"***** Response from calling {self.role} ({id}) *****"
            f(colored(tool_print, "green"), flush=True)
            # f(self.content, flush=True)
            display(Markdown(self.content))
            f(colored("*" * len(tool_print), "green"), flush=True)
        else:
            # f(self.content, flush=True)
            display(Markdown(self.content))


@wrap_message
class ToolResponseMessage(BasePrintReceivedMessage):
    role: MessageRole = "tool"
    tool_responses: list[ToolResponse]
    content: Union[str, int, float, bool]

    def print(self, f: Optional[Callable[..., Any]] = None) -> None:
        f = f or print
        super().print(f)

        for tool_response in self.tool_responses:
            tool_response.print(f)
            if cmbagent_debug:
                f("\n", "-" * 80, flush=True, sep="")


class FunctionCall(BaseModel):
    name: Optional[str] = None
    arguments: Optional[str] = None

    def print(self, f: Optional[Callable[..., Any]] = None) -> None:
        f = f or print

        name = self.name or "(No function name found)"
        arguments = self.arguments or "(No arguments found)"

        func_print = f"***** Suggested function call: {name} *****"
        f(colored(func_print, "green"), flush=True)
        f(
            "Arguments: \n",
            arguments,
            flush=True,
            sep="",
        )
        f(colored("*" * len(func_print), "green"), flush=True)


@wrap_message
class FunctionCallMessage(BasePrintReceivedMessage):
    content: Optional[Union[str, int, float, bool]] = None  # type: ignore [assignment]
    function_call: FunctionCall

    def print(self, f: Optional[Callable[..., Any]] = None) -> None:
        f = f or print
        super().print(f)

        if self.content is not None:
            f(self.content, flush=True)

        self.function_call.print(f)

        if cmbagent_debug:
            f("\n", "-" * 80, flush=True, sep="")


class ToolCall(BaseModel):
    id: Optional[str] = None
    function: FunctionCall
    type: str

    def print(self, f: Optional[Callable[..., Any]] = None) -> None:
        f = f or print

        id = self.id or "No tool call id found"

        name = self.function.name or "(No function name found)"
        arguments = self.function.arguments or "(No arguments found)"

        if cmbagent_debug:
            func_print = f"***** Suggested tool call ({id}): {name} *****"
            f(colored(func_print, "green"), flush=True)
            f(
                "Arguments: \n",
                arguments,
                flush=True,
                sep="",
            )
            f(colored("*" * len(func_print), "green"), flush=True)


@wrap_message
class ToolCallMessage(BasePrintReceivedMessage):
    content: Optional[Union[str, int, float, bool]] = None  # type: ignore [assignment]
    refusal: Optional[str] = None
    role: Optional[MessageRole] = None
    audio: Optional[str] = None
    function_call: Optional[FunctionCall] = None
    tool_calls: list[ToolCall]

    def print(self, f: Optional[Callable[..., Any]] = None) -> None:
        f = f or print
        super().print(f)

        if self.content is not None:
            if self.sender_name in [#"plan_reviewer",
                                    "reviewer_response_formatter",
                                    "planner_response_formatter",
                                    "engineer_response_formatter", 
                                    "control",
                                    "camels_agent",
                                    "admin",
                                    "camels_response_formatter",
                                    "classy_sz_response_formatter",
                                    "joke_critique_response_formatter",
                                    "joker_response_formatter",
                                    "lecturer_response_formatter",
                                    "course_director_response_formatter",
                                    "course_material_provider"
                                    ]:
                display(Markdown(self.content)) # it doesnt work all the time
            else:
                f(self.content, flush=True)
            # cmbagent debug
            # print("\n in agent_messages.py ToolCallMessage print... cmbagent debug")
            # display(Markdown(self.content)) # it doesnt work all the time

        for tool_call in self.tool_calls:
            # print('\n\n in agent_messages.py ToolCallMessage print... tool_call: ', tool_call)
            tool_call.print(f)

        if cmbagent_debug:
            f("\n", "-" * 80, flush=True, sep="")


@wrap_message
class TextMessage(BasePrintReceivedMessage):
    content: Optional[Union[str, int, float, bool, list[dict[str, Union[str, dict[str, Any]]]]]] = None  # type: ignore [assignment]

    def print(self, f: Optional[Callable[..., Any]] = None) -> None:
        f = f or print
        super().print(f)

        if self.content is not None:
            # original code
            # f(content_str(self.content), flush=True)  # type: ignore [arg-type] 
            # print("\n content: ", self.content)
            # print("\n in agent_messages.py TextMessage print... sender_name: ", self.sender_name)
            # cmbagent debug
            # print("\n in agent_messages.py TextMessage print... cmbagent debug")
            if self.sender_name in [#"plan_reviewer",
                                    "reviewer_response_formatter",
                                    "planner_response_formatter",
                                    "engineer_response_formatter", 
                                    "control",
                                    "camels_agent",
                                    "admin",
                                    "camels_response_formatter",
                                    # "researcher_response_formatter",
                                    "classy_sz_response_formatter",
                                    "joke_critique_response_formatter",
                                    "joker_response_formatter",
                                    "lecturer_response_formatter",
                                    "course_director_response_formatter",
                                    "course_material_provider",
                                    "review_recorder",
                                    # "planner",
                                    # "plan_reviewer",
                                    ]:
                display(Markdown(self.content)) # it doesnt work all the time
            elif self.sender_name in ["classy_sz_agent",
                                      "camels_agent",
                                      "engineer",
                                      "researcher",
                                      "planner",
                                      "plan_reviewer",
                                      "joker",
                                      "joke_critique",
                                      "lecturer",
                                      "course_director"]:
                display(Markdown("\nForwarding content for formatting...\n"))
                if cmbagent_debug:
                    # if self.sender_name == "engineer":
                    print('in agent_messages.py TextMessage print... self.sender_name: ', self.sender_name)
                    print('in agent_messages.py TextMessage print... self.content: ', self.content)
                    # import sys; sys.exit()

            elif self.sender_name in ["structured_code_agent"]:
                display(Markdown("\nForwarding to executor...\n"))
                f(content_str(self.content), flush=True)
            else:
                f(content_str(self.content), flush=True)  # type: ignore [arg-type]

        if cmbagent_debug:
            f("\n", "-" * 80, flush=True, sep="")


def create_received_message_model(
    *, uuid: Optional[UUID] = None, message: dict[str, Any], sender: "Agent", recipient: "Agent"
) -> Union[FunctionResponseMessage, ToolResponseMessage, FunctionCallMessage, ToolCallMessage, TextMessage]:
    # print(f"{message=}")
    # print(f"{sender=}")

    role = message.get("role")
    if role == "function":
        return FunctionResponseMessage(**message, sender_name=sender.name, recipient_name=recipient.name, uuid=uuid)
    if role == "tool":
        return ToolResponseMessage(**message, sender_name=sender.name, recipient_name=recipient.name, uuid=uuid)

    # Role is neither function nor tool

    if message.get("function_call"):
        return FunctionCallMessage(
            **message,
            sender_name=sender.name,
            recipient_name=recipient.name,
            uuid=uuid,
        )

    if message.get("tool_calls"):
        return ToolCallMessage(
            **message,
            sender_name=sender.name,
            recipient_name=recipient.name,
            uuid=uuid,
        )

    # Now message is a simple content message
    content = message.get("content")
    allow_format_str_template = (
        recipient.llm_config.get("allow_format_str_template", False) if recipient.llm_config else False  # type: ignore [attr-defined]
    )
    if content is not None and "context" in message:
        content = OpenAIWrapper.instantiate(
            content,  # type: ignore [arg-type]
            message["context"],
            allow_format_str_template,
        )

    return TextMessage(
        content=content,
        sender_name=sender.name,
        recipient_name=recipient.name,
        uuid=uuid,
    )


@wrap_message
class PostCarryoverProcessingMessage(BaseMessage):
    carryover: Union[str, list[Union[str, dict[str, Any], Any]]]
    message: str
    verbose: bool = False

    sender_name: str
    recipient_name: str
    summary_method: str
    summary_args: Optional[dict[str, Any]] = None
    max_turns: Optional[int] = None

    def __init__(self, *, uuid: Optional[UUID] = None, chat_info: dict[str, Any]):
        carryover = chat_info.get("carryover", "")
        message = chat_info.get("message")
        verbose = chat_info.get("verbose", False)

        sender_name = chat_info["sender"].name
        recipient_name = chat_info["recipient"].name
        summary_args = chat_info.get("summary_args")
        max_turns = chat_info.get("max_turns")

        # Fix Callable in chat_info
        summary_method = chat_info.get("summary_method", "")
        if callable(summary_method):
            summary_method = summary_method.__name__

        print_message = ""
        if isinstance(message, str):
            print_message = message
        elif callable(message):
            print_message = "Callable: " + message.__name__
        elif isinstance(message, dict):
            print_message = "Dict: " + str(message)
        elif message is None:
            print_message = "None"

        super().__init__(
            uuid=uuid,
            carryover=carryover,
            message=print_message,
            verbose=verbose,
            summary_method=summary_method,
            summary_args=summary_args,
            max_turns=max_turns,
            sender_name=sender_name,
            recipient_name=recipient_name,
        )

    def _process_carryover(self) -> str:
        if not isinstance(self.carryover, list):
            return self.carryover

        print_carryover = []
        for carryover_item in self.carryover:
            if isinstance(carryover_item, str):
                print_carryover.append(carryover_item)
            elif isinstance(carryover_item, dict) and "content" in carryover_item:
                print_carryover.append(str(carryover_item["content"]))
            else:
                print_carryover.append(str(carryover_item))

        return ("\n").join(print_carryover)

    def print(self, f: Optional[Callable[..., Any]] = None) -> None:
        f = f or print

        print_carryover = self._process_carryover()

        f(colored("\n" + "*" * 80, "blue"), flush=True, sep="")
        f(
            colored(
                "Starting a new chat....",
                "blue",
            ),
            flush=True,
        )
        if self.verbose:
            f(colored("Message:\n" + self.message, "blue"), flush=True)
            f(colored("Carryover:\n" + print_carryover, "blue"), flush=True)
        f(colored("\n" + "*" * 80, "blue"), flush=True, sep="")


@wrap_message
class ClearAgentsHistoryMessage(BaseMessage):
    agent_name: Optional[str] = None
    nr_messages_to_preserve: Optional[int] = None

    def __init__(
        self,
        *,
        uuid: Optional[UUID] = None,
        agent: Optional["Agent"] = None,
        nr_messages_to_preserve: Optional[int] = None,
    ):
        return super().__init__(
            uuid=uuid, agent_name=agent.name if agent else None, nr_messages_to_preserve=nr_messages_to_preserve
        )

    def print(self, f: Optional[Callable[..., Any]] = None) -> None:
        f = f or print

        if self.agent_name:
            if self.nr_messages_to_preserve:
                f(f"Clearing history for {self.agent_name} except last {self.nr_messages_to_preserve} messages.")
            else:
                f(f"Clearing history for {self.agent_name}.")
        else:
            if self.nr_messages_to_preserve:
                f(f"Clearing history for all agents except last {self.nr_messages_to_preserve} messages.")
            else:
                f("Clearing history for all agents.")


# todo: break into multiple messages
@wrap_message
class SpeakerAttemptSuccessfulMessage(BaseMessage):
    mentions: dict[str, int]
    attempt: int
    attempts_left: int
    verbose: Optional[bool] = False

    def __init__(
        self,
        *,
        uuid: Optional[UUID] = None,
        mentions: dict[str, int],
        attempt: int,
        attempts_left: int,
        select_speaker_auto_verbose: Optional[bool] = False,
    ):
        super().__init__(
            uuid=uuid,
            mentions=deepcopy(mentions),
            attempt=attempt,
            attempts_left=attempts_left,
            verbose=select_speaker_auto_verbose,
        )

    def print(self, f: Optional[Callable[..., Any]] = None) -> None:
        f = f or print

        selected_agent_name = next(iter(self.mentions))
        f(
            colored(
                f">>>>>>>> Select speaker attempt {self.attempt} of {self.attempt + self.attempts_left} successfully selected: {selected_agent_name}",
                "green",
            ),
            flush=True,
        )


@wrap_message
class SpeakerAttemptFailedMultipleAgentsMessage(BaseMessage):
    mentions: dict[str, int]
    attempt: int
    attempts_left: int
    verbose: Optional[bool] = False

    def __init__(
        self,
        *,
        uuid: Optional[UUID] = None,
        mentions: dict[str, int],
        attempt: int,
        attempts_left: int,
        select_speaker_auto_verbose: Optional[bool] = False,
    ):
        super().__init__(
            uuid=uuid,
            mentions=deepcopy(mentions),
            attempt=attempt,
            attempts_left=attempts_left,
            verbose=select_speaker_auto_verbose,
        )

    def print(self, f: Optional[Callable[..., Any]] = None) -> None:
        f = f or print

        f(
            colored(
                f">>>>>>>> Select speaker attempt {self.attempt} of {self.attempt + self.attempts_left} failed as it included multiple agent names.",
                "red",
            ),
            flush=True,
        )


@wrap_message
class SpeakerAttemptFailedNoAgentsMessage(BaseMessage):
    mentions: dict[str, int]
    attempt: int
    attempts_left: int
    verbose: Optional[bool] = False

    def __init__(
        self,
        *,
        uuid: Optional[UUID] = None,
        mentions: dict[str, int],
        attempt: int,
        attempts_left: int,
        select_speaker_auto_verbose: Optional[bool] = False,
    ):
        super().__init__(
            uuid=uuid,
            mentions=deepcopy(mentions),
            attempt=attempt,
            attempts_left=attempts_left,
            verbose=select_speaker_auto_verbose,
        )

    def print(self, f: Optional[Callable[..., Any]] = None) -> None:
        f = f or print

        f(
            colored(
                f">>>>>>>> Select speaker attempt #{self.attempt} failed as it did not include any agent names.",
                "red",
            ),
            flush=True,
        )


@wrap_message
class GroupChatResumeMessage(BaseMessage):
    last_speaker_name: str
    messages: list[dict[str, Any]]
    verbose: Optional[bool] = False

    def __init__(
        self,
        *,
        uuid: Optional[UUID] = None,
        last_speaker_name: str,
        messages: list[dict[str, Any]],
        silent: Optional[bool] = False,
    ):
        super().__init__(uuid=uuid, last_speaker_name=last_speaker_name, messages=messages, verbose=not silent)

    def print(self, f: Optional[Callable[..., Any]] = None) -> None:
        f = f or print

        f(
            f"Prepared group chat with {len(self.messages)} messages, the last speaker is",
            colored(self.last_speaker_name, "yellow"),
            flush=True,
        )


@wrap_message
class GroupChatRunChatMessage(BaseMessage):
    speaker_name: str
    verbose: Optional[bool] = False

    def __init__(self, *, uuid: Optional[UUID] = None, speaker: "Agent", silent: Optional[bool] = False):
        super().__init__(uuid=uuid, speaker_name=speaker.name, verbose=not silent)

    def print(self, f: Optional[Callable[..., Any]] = None) -> None:
        f = f or print

        if cmbagent_debug:
            f(colored(f"\nCalling: {self.speaker_name}...\n", "green"), flush=True)
        else:
            if self.speaker_name not in ["_Swarm_Tool_Executor"]:
                color = cmbagent_color_dict.get(self.speaker_name, cmbagent_default_color)
                f(colored(f"\nCalling {self.speaker_name}...\n", color), flush=True)


@wrap_message
class TerminationAndHumanReplyMessage(BaseMessage):
    no_human_input_msg: str
    sender_name: str
    recipient_name: str

    def __init__(
        self,
        *,
        uuid: Optional[UUID] = None,
        no_human_input_msg: str,
        sender: Optional["Agent"] = None,
        recipient: "Agent",
    ):
        super().__init__(
            uuid=uuid,
            no_human_input_msg=no_human_input_msg,
            sender_name=sender.name if sender else "No sender",
            recipient_name=recipient.name,
        )

    def print(self, f: Optional[Callable[..., Any]] = None) -> None:
        f = f or print

        f(colored(f"\n>>>>>>>> {self.no_human_input_msg}", "red"), flush=True)


@wrap_message
class UsingAutoReplyMessage(BaseMessage):
    human_input_mode: str
    sender_name: str
    recipient_name: str

    def __init__(
        self,
        *,
        uuid: Optional[UUID] = None,
        human_input_mode: str,
        sender: Optional["Agent"] = None,
        recipient: "Agent",
    ):
        super().__init__(
            uuid=uuid,
            human_input_mode=human_input_mode,
            sender_name=sender.name if sender else "No sender",
            recipient_name=recipient.name,
        )

    def print(self, f: Optional[Callable[..., Any]] = None) -> None:
        f = f or print
        if cmbagent_debug:
            f(colored("\n>>>>>>>> USING AUTO REPLY...", "red"), flush=True)


@wrap_message
class ExecuteCodeBlockMessage(BaseMessage):
    code: str
    language: str
    code_block_count: int
    recipient_name: str

    def __init__(
        self, *, uuid: Optional[UUID] = None, code: str, language: str, code_block_count: int, recipient: "Agent"
    ):
        super().__init__(
            uuid=uuid, code=code, language=language, code_block_count=code_block_count, recipient_name=recipient.name
        )

    def print(self, f: Optional[Callable[..., Any]] = None) -> None:
        f = f or print
        if cmbagent_debug:
            f(
                colored(
                    f"\n>>>>>>>> EXECUTING CODE BLOCK {self.code_block_count} (inferred language is {self.language})...",
                    "red",
                ),
                flush=True,
            )


@wrap_message
class ExecuteFunctionMessage(BaseMessage):
    func_name: str
    call_id: Optional[str] = None
    arguments: dict[str, Any]
    recipient_name: str

    def __init__(
        self,
        *,
        uuid: Optional[UUID] = None,
        func_name: str,
        call_id: Optional[str] = None,
        arguments: dict[str, Any],
        recipient: "Agent",
    ):
        super().__init__(
            uuid=uuid, func_name=func_name, call_id=call_id, arguments=arguments, recipient_name=recipient.name
        )

    def print(self, f: Optional[Callable[..., Any]] = None) -> None:
        f = f or print

        if cmbagent_debug:
            f(
                colored(
                    f"\n>>>>>>>> EXECUTING FUNCTION {self.func_name}...\nCall ID: {self.call_id}\nInput arguments: {self.arguments}",
                    "magenta",
                ),
                flush=True,
            )
 


@wrap_message
class ExecutedFunctionMessage(BaseMessage):
    func_name: str
    call_id: Optional[str] = None
    arguments: dict[str, Any]
    content: str
    recipient_name: str

    def __init__(
        self,
        *,
        uuid: Optional[UUID] = None,
        func_name: str,
        call_id: Optional[str] = None,
        arguments: dict[str, Any],
        content: str,
        recipient: "Agent",
    ):
        super().__init__(
            uuid=uuid,
            func_name=func_name,
            call_id=call_id,
            arguments=arguments,
            content=content,
            recipient_name=recipient.name,
        )

    def print(self, f: Optional[Callable[..., Any]] = None) -> None:
        f = f or print

        f(
            colored(
                f"\n>>>>>>>> EXECUTED FUNCTION {self.func_name}...\nCall ID: {self.call_id}\nInput arguments: {self.arguments}\nOutput:\n{self.content}",
                "magenta",
            ),
            flush=True,
        )


@wrap_message
class SelectSpeakerMessage(BaseMessage):
    agent_names: Optional[list[str]] = None

    def __init__(self, *, uuid: Optional[UUID] = None, agents: Optional[list["Agent"]] = None):
        agent_names = [agent.name for agent in agents] if agents else None
        super().__init__(uuid=uuid, agent_names=agent_names)

    def print(self, f: Optional[Callable[..., Any]] = None) -> None:
        f = f or print

        f("Please select the next speaker from the following list:")
        agent_names = self.agent_names or []
        for i, agent_name in enumerate(agent_names):
            f(f"{i + 1}: {agent_name}")


@wrap_message
class SelectSpeakerTryCountExceededMessage(BaseMessage):
    try_count: int
    agent_names: Optional[list[str]] = None

    def __init__(self, *, uuid: Optional[UUID] = None, try_count: int, agents: Optional[list["Agent"]] = None):
        agent_names = [agent.name for agent in agents] if agents else None
        super().__init__(uuid=uuid, try_count=try_count, agent_names=agent_names)

    def print(self, f: Optional[Callable[..., Any]] = None) -> None:
        f = f or print

        f(f"You have tried {self.try_count} times. The next speaker will be selected automatically.")


@wrap_message
class SelectSpeakerInvalidInputMessage(BaseMessage):
    agent_names: Optional[list[str]] = None

    def __init__(self, *, uuid: Optional[UUID] = None, agents: Optional[list["Agent"]] = None):
        agent_names = [agent.name for agent in agents] if agents else None
        super().__init__(uuid=uuid, agent_names=agent_names)

    def print(self, f: Optional[Callable[..., Any]] = None) -> None:
        f = f or print

        f(f"Invalid input. Please enter a number between 1 and {len(self.agent_names or [])}.")


@wrap_message
class ClearConversableAgentHistoryMessage(BaseMessage):
    agent_name: str
    recipient_name: str
    no_messages_preserved: int

    def __init__(self, *, uuid: Optional[UUID] = None, agent: "Agent", no_messages_preserved: Optional[int] = None):
        super().__init__(
            uuid=uuid,
            agent_name=agent.name,
            recipient_name=agent.name,
            no_messages_preserved=no_messages_preserved,
        )

    def print(self, f: Optional[Callable[..., Any]] = None) -> None:
        f = f or print

        for _ in range(self.no_messages_preserved):
            f(
                f"Preserving one more message for {self.agent_name} to not divide history between tool call and "
                f"tool response."
            )


@wrap_message
class ClearConversableAgentHistoryWarningMessage(BaseMessage):
    recipient_name: str

    def __init__(self, *, uuid: Optional[UUID] = None, recipient: "Agent"):
        super().__init__(
            uuid=uuid,
            recipient_name=recipient.name,
        )

    def print(self, f: Optional[Callable[..., Any]] = None) -> None:
        f = f or print

        f(
            colored(
                "WARNING: `nr_preserved_messages` is ignored when clearing chat history with a specific agent.",
                "yellow",
            ),
            flush=True,
        )


@wrap_message
class GenerateCodeExecutionReplyMessage(BaseMessage):
    code_block_languages: list[str]
    sender_name: Optional[str] = None
    recipient_name: str

    def __init__(
        self,
        *,
        uuid: Optional[UUID] = None,
        code_blocks: list["CodeBlock"],
        sender: Optional["Agent"] = None,
        recipient: "Agent",
    ):
        code_block_languages = [code_block.language for code_block in code_blocks]

        super().__init__(
            uuid=uuid,
            code_block_languages=code_block_languages,
            sender_name=sender.name if sender else None,
            recipient_name=recipient.name,
        )

    def print(self, f: Optional[Callable[..., Any]] = None) -> None:
        f = f or print

        num_code_blocks = len(self.code_block_languages)
        if cmbagent_debug:
            if num_code_blocks == 1:
                f(
                    colored(
                        f"\n>>>>>>>> EXECUTING CODE BLOCK (inferred language is {self.code_block_languages[0]})...",
                        "red",
                    ),
                    flush=True,
                )
            else:
                f(
                    colored(
                        f"\n>>>>>>>> EXECUTING {num_code_blocks} CODE BLOCKS (inferred languages are [{', '.join([x for x in self.code_block_languages])}])...",
                        "red",
                    ),
                    flush=True,
                )


@wrap_message
class ConversableAgentUsageSummaryNoCostIncurredMessage(BaseMessage):
    recipient_name: str

    def __init__(self, *, uuid: Optional[UUID] = None, recipient: "Agent"):
        super().__init__(uuid=uuid, recipient_name=recipient.name)

    def print(self, f: Optional[Callable[..., Any]] = None) -> None:
        f = f or print

        f(f"No cost incurred from agent '{self.recipient_name}'.")


@wrap_message
class ConversableAgentUsageSummaryMessage(BaseMessage):
    recipient_name: str

    def __init__(self, *, uuid: Optional[UUID] = None, recipient: "Agent"):
        super().__init__(uuid=uuid, recipient_name=recipient.name)

    def print(self, f: Optional[Callable[..., Any]] = None) -> None:
        f = f or print

        f(f"Agent '{self.recipient_name}':")
