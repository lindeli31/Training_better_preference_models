from typing import List

from src.core.Message import Message
from src.core.PermPair import PermPair
from src.core.Prompt import Prompt
from src.core.Template import Template


class GepaApertus8bTemplate(Template):
    generation_instructions: str = """
Evaluate and compare two response candidates to determine which one better addresses the user's question or topic, considering various domains such as international Chinese education, language governance, language ethics, astrology, self-development, React Native development, Salesforce Commerce Cloud integration, encryption, security, date and time handling, PHP, PowerShell, Python, Ruby, and other technical and non-technical areas.

The evaluation should be based on a scale from -2 (the first answer is much better) to +2 (the second answer is much better). Return -1 if the first answer candidate is better, and +1 if the second answer candidate is better. Return 0 if they are equal.

When evaluating responses, consider the following key factors:
1. **Relevance**: How well does the answer address the user's question or topic?
2. **Accuracy**: How accurate is the information provided in the answer?
3. **Completeness**: How comprehensive is the answer in covering the topic or question?
4. **Clarity**: How easy is the answer to understand?
5. **Organization**: How well-structured and organized is the answer?
6. **Domain Knowledge**: Consider the specific domain or topic of the conversation, such as:
   - **International Chinese Education**: Evaluate the author's credentials, publication dates, and journal names when assessing answers that provide lists of research papers or academic sources.
   - **Astrology**: Consider the astrological signs, planetary transits, and their potential impacts on individuals, as well as the provision of personalized advice or recommendations based on astrological principles.
   - **Self-Development and Motivation**: Assess the answers based on their ability to provide inspiring and practical advice, introduce relevant books or resources, and offer strategies for personal growth and goal achievement.
   - **React Native Development**: Consider the use of appropriate libraries (e.g., `react-native-ble-plx` for Bluetooth functionality), correct implementation of features (e.g., scanning for nearby Bluetooth devices when the app is idle), and adherence to best practices for coding and error handling.
   - **Salesforce Commerce Cloud Integration**: Evaluate the use of correct API endpoints, authentication methods, and handling of sensitive data. Consider the ability to manage orders, customers, and other resources as per the Salesforce Commerce Cloud API documentation.
   - **Encryption and Security**: Consider the use of secure encryption algorithms (e.g., Blowfish), modes of operation (e.g., CBC), and key management practices.
   - **Date and Time Handling**: Consider the use of libraries like CarbonImmutable for parsing and manipulating dates, and the importance of timezone information and handling daylight saving time (DST) correctly.
   - **PHP**: Consider the use of `fscanf`, `fgets`, and `explode` functions to read input, and the `echo` statement to print output. Evaluate the accuracy and completeness of the information provided.
   - **PowerShell**: Consider the use of `Get-ChildItem` cmdlet to search for files, and the `Write-Host` cmdlet to print output. Evaluate the use of error handling and validation to ensure input is valid and in the expected format.
   - **Python**: Consider the use of `print` function to print output, and the importance of error handling and validation to ensure input is valid and in the expected format.
   - **Ruby**: Consider the use of `puts` method to print output, and the importance of error handling and validation to ensure input is valid and in the expected format.

When evaluating code snippets, consider the correctness of the code, its readability, and whether it addresses the user's question directly. Also, assess whether the code is well-structured, follows best practices, and includes proper error handling.

In the context of encryption, consider the importance of using secure encryption algorithms, generating random initialization vectors (IVs), and managing encryption keys securely. For date and time handling, consider the importance of using timezone information and handling DST correctly.

When evaluating answers related to React Native development, consider the use of appropriate libraries, correct implementation of features, and adherence to best practices for coding and error handling. For Salesforce Commerce Cloud integration, consider the use of correct API endpoints, authentication methods, and handling of sensitive data.

In the context of self-development and motivation, consider the ability of the answer to provide inspiring and practical advice, introduce relevant books or resources, and offer strategies for personal growth and goal achievement. For astrology, consider the provision of personalized advice or recommendations based on astrological principles.

In the context of international Chinese education, consider the author's credentials, publication dates, and journal names when evaluating answers that provide lists of research papers or academic sources.

When evaluating HTML, CSS, and JavaScript code, consider the correctness of the code, its readability, and whether it addresses the user's question directly. Also, assess whether the code is well-structured, follows best practices, and includes proper error handling.

In general, consider the following strategies when evaluating responses:
- Look for answers that provide clear and concise explanations.
- Evaluate the accuracy and completeness of the information provided.
- Consider the relevance of the answer to the user's question or topic.
- Assess the organization and structure of the answer.
- Consider the use of proper terminology and domain-specific knowledge.
- Evaluate the ability of the answer to provide practical advice or solutions.
- Consider the importance of security, encryption, and data handling in the context of the question.

When comparing response candidates, consider the following:
- If one answer provides more accurate or complete information, prefer that answer.
- If one answer is more relevant or better organized, prefer that answer.
- If one answer provides more practical advice or solutions, prefer that answer.
- If one answer demonstrates better domain-specific knowledge, prefer that answer.

By following these guidelines, you can provide accurate and consistent evaluations of response candidates, considering various domains and technical areas.
""".strip()
    
    def render(self, perm_pair: PermPair) -> List[Prompt]:
        return [
            self._from_perm_to_prompt(perm_pair.context, perm_pair.answer_candidate_a, perm_pair.answer_candidate_b, perm_pair.metadata),
            self._from_perm_to_prompt(perm_pair.context, perm_pair.answer_candidate_b, perm_pair.answer_candidate_a, perm_pair.metadata),
        ]

    def _from_perm_to_prompt(
            self,
            context,
            answer_candidate_a,
            answer_candidate_b,
            metadata,
    ) -> Prompt:
        system_message = (
            Message(role="system")
            .add_text(self.generation_instructions)
        )

        user_message = (
            Message(role="user")
            .add_context(context)
            .add_text("RESPONSES:\n")
            .add_text(f"answer candidate A: {answer_candidate_a}\n")
            .add_text(f"answer candidate B: {answer_candidate_b}\n")
            .add_text("ANSWER:") # should we keep it?
        )

        return Prompt(messages=[system_message, user_message])
