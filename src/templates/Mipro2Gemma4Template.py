from typing import List

from src.core.Message import Message
from src.core.PermPair import PermPair
from src.core.Prompt import Prompt
from src.core.Template import Template


class Mipro2Gemma4Template(Template):
    generation_instructions: str = """
You are an impartial judge evaluating two potential AI-generated responses to a conversation. Your goal is to determine which response better satisfies the user's request based on accuracy, relevance, and helpfulness.

**Evaluation Scale:**
- `-2`: Answer Candidate A is much better.
- `-1`: Answer Candidate A is better.
- `0`: Both answers are equal in quality.
- `+1`: Answer Candidate B is better.
- `+2`: Answer Candidate B is much better.

**Requirements:**
1. Review the provided conversation history to understand the context and the user's intent.
2. Compare Answer Candidate A and Answer Candidate B.
3. Your judgment must be objective and consistent, regardless of the order in which the candidates are presented.
4. Provide your final decision as a single integer from the scale above.

**Output:**
Return only the numerical score.

**Examples:**

Example 1:
"conversation_history": "Message from: user\nЯ хочу написать пост в твиттер о том, что моя опенсурс библиотека ai-commits пока до сих пор единственная, в которой есть семантические коммиты и Gitmoji. Коммиты получаются красивыми. \n\nMessage from: assistant\nЗвучит здорово! Вот пример того, как вы можете написать этот пост в Твиттере:\n\n\"Продолжаю работать над своей открытой библиотекой @ai_commits и рад сообщить, что это по-прежнему единственная библиотека, которая поддерживает семантические коммиты и Gitmoji! Результат - красивые и понятные коммиты. Попробуйте сейчас же! #ai_commits #opensource #gitmoji\"\n\nMessage from: user\nМне нравится 3 вариант, но туда стоит добавить то, что коммиты генерируются автоматически с помощью ChatGPT API\n\nMessage from: assistant\nОтлично! Вот как можно изменить пост, чтобы включить эту информацию:\n\n\"Продолжаю работать над своей открытой библиотекой @ai_commits и рад сообщить, что это по-прежнему единственная библиотека, которая поддерживает семантические коммиты и Gitmoji! Результат - красивые и понятные коммиты, которые генерируются автоматически с помощью ChatGPT API. Попробуйте сейчас же! #ai_commits #opensource #gitmoji #chatgpt\"\n\nMessage from: user\nПереведи на английский и добавь, что она полностью бесплатна (если есть OpenAI API Key)\n\n",
"answer_candidate_a": "Sure, here's the revised post in English:\n\n\"I'm happy to share that my open-source library @ai_commits is still the only one that supports semantic commits and Gitmoji! The result is beautiful and readable commits, generated automatically using ChatGPT API. And the best part? It's completely free if you have an OpenAI API key. Try it now! #ai_commits #opensource #gitmoji #chatgpt\"",
"answer_candidate_b": "Here's the translated and updated post:\n\n\"Continuing to work on my open-source library @ai_commits, I'm excited to share that it remains the only library supporting semantic commits and Gitmoji! The result is beautiful and understandable commits generated automatically using the ChatGPT API. Try it now! #ai_commits #open_source #gitmoji #chatgpt (Note: It's completely free if you have an OpenAI API Key)\"",
"preference": "-1"

Example 2:
"conversation_history": "Message from: user\nwrite a java code to remove a document in firestore using where string equals sometext\n\nMessage from: assistant\n // Function to remove a document by title\n  public void removeDocument(String documentTitle) {\n    DocumentReference docRef = collectionReference.document(documentTitle);\n    docRef.delete().addOnSuccessListener(aVoid -> {\n      System.out.println(\"Document removed successfully\");\n    }).addOnFailureListener(e -> {\n      System.out.println(\"Error removing document: \" + e.getMessage());\n    });\n  }\n\nMessage from: user\nwrite a java code to remove a document in firestore using where string equals sometext using com.google.firebase.cloud.FirestoreClient\n\nMessage from: assistant\n // Function to remove a document by title\n  public void removeDocument(String documentTitle) {\n    DocumentReference docRef = collectionReference.document(documentTitle);\n    docRef.delete().addOnSuccessListener(aVoid -> {\n      System.out.println(\"Document removed successfully\");\n    }).addOnFailureListener(e -> {\n      System.out.println(\"Error removing document: \" + e.getMessage());\n    });\n  }\n\n  // Function to remove all documents in a collection\n  public void removeAllDocuments() {\n    collectionReference.delete();\n  }\n}\n\nMessage from: user\nThank you so much\n\n",
"answer_candidate_a": " You're welcome! I'm glad I could help.\n\nTo give you a more complete example, here's how you can use `FirestoreClient` to remove documents based on a specific field value using a query:\n\n```java\nimport com.google.cloud.firestore.Firestore;\nimport com.google.cloud.firestore.FirestoreOptions;\nimport com.google.cloud.firestore.QueryDocumentSnapshot;\nimport com.google.cloud.firestore.QuerySnapshot;\n\npublic class FirestoreExample {\n\n    private static final String PROJECT_ID = \"your-project-id\";\n    private static final String COLLECTION_NAME = \"your-collection-name\";\n    private static final String FIELD_NAME = \"your-field-name\";\n    private static final String FIELD_VALUE = \"some-text\";\n\n    public static void main(String[] args) throws Exception {\n        // Initialize Firestore\n        FirestoreOptions firestoreOptions =\n                FirestoreOptions.getDefaultInstance().toBuilder()\n                        .setProjectId(PROJECT_ID)\n                        .build();\n        Firestore firestore = firestoreOptions.getService();\n\n        // Remove documents where field equals some text\n        removeDocumentsByFieldValue(firestore, COLLECTION_NAME, FIELD_NAME, FIELD_VALUE);\n    }\n\n    public static void removeDocumentsByFieldValue(Firestore firestore, String collectionName, String fieldName, String fieldValue) throws Exception {\n        // Create a query against the collection\n        QuerySnapshot querySnapshot = firestore.collection(collectionName)\n                .whereEqualTo(fieldName, fieldValue)\n                .get()\n                .get();\n\n        // Iterate through the documents and delete them\n        for (QueryDocumentSnapshot document : querySnapshot) {\n            document.getReference().delete();\n            System.out.println(\"Document removed: \" + document.getId());\n        }\n    }\n}\n```\n\n### Explanation:\n1. **Initialize Firestore**: Set up the Firestore client with your project ID.\n2. **Query Documents**: Use a query to find documents where a specific field equals a given value.\n3. **Delete Documents**: Iterate through the query results and delete each document.\n\n### Dependencies:\nMake sure you have the Firestore dependency in your `pom.xml` if you're using Maven:\n\n```xml\n<dependency>\n    <groupId>com.google.cloud</groupId>\n    <artifactId>google-cloud-firestore</artifactId>\n    <version>2.1.0</version>\n</dependency>\n```\n\nOr in your `build.gradle` if you're using Gradle:\n\n```groovy\nimplementation 'com.google.cloud:google-cloud-firestore:2.1.0'\n```\n\nThis code will remove all documents in the specified collection where the specified field equals the given value. Make sure to replace `your-project-id`, `your-collection-name`, `your-field-name`, and `some-text` with your actual values.",
"answer_candidate_b": "You're welcome! If you have any other questions, feel free to ask. I'm here to help.",
"preference": "+1"
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
