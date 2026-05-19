from typing import List

from src.core.Message import Message
from src.core.PermPair import PermPair
from src.core.Prompt import Prompt
from src.core.Template import Template


class Mipro4Gemma4Template(Template):
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

Example 3:
"conversation_history": "Message from: user\n一道牛肉炖土豆的背后，需要多少人的努力？\n\nMessage from: assistant\n一道牛肉炖土豆的背后，需要多方人员的努力，包括：\n\n1. **养牛人**：负责牛的饲养，保证牛肉的质量。\n2. **屠宰场工人**：负责牛的屠宰，并将牛肉分割、包装。\n3. **运输人员**：负责牛肉从屠宰场运送到市场或超市。\n4. **市场或超市工作人员**：负责牛肉的销售。\n5. **采购员**：负责土豆的采购，土豆的种植需要农民的辛勤劳作。\n6. **运输人员**：负责土豆从农田运送到市场或超市。\n7. **厨师或家庭主妇**：负责牛肉炖土豆的烹饪。\n8. **水电工人**：负责提供水电等基础设施，保证牛肉炖土豆的烹饪条件。\n9. **环保工作者**：负责环境保护，确保食品安全。\n\n因此，一道牛肉炖土豆的背后，需要多方人员的共同努力。\n\nMessage from: user\n如果增加施肥，除草，车辆制造的环节呢？\n\nMessage from: assistant\n如果考虑到土豆种植和牛肉运输等环节的更多细节，那么还需要增加以下人员的努力：\n\n1. **土豆种植环节**：\n\t* **种植户**：负责土豆的种植。\n\t* **肥料生产商**：负责肥料的生产，为土豆提供营养。\n\t* **除草剂生产商**：负责除草剂的生产，帮助控制杂草。\n\t* **农机操作手**：负责使用农机进行播种、施肥、除草等作业。\n2. **牛肉运输环节**：\n\t* **车辆制造商**：负责运输车辆的制造，保证牛肉运输的顺利进行。\n\t* **维修人员**：负责运输车辆的维修保养，确保其正常运行。\n\n因此，如果考虑到更多的细节环节，那么一道牛肉炖土豆的背后，需要更多的人员共同努力，包括种植户、肥料生产商、除草剂生产商、农机操作手、车辆制造商、维修人员等。\n\nMessage from: user\n请继续拆分环节，尽可能的拆分细致，比如制造机械需要的冶炼环节，运输需要先修路的环节，穷尽你知道的环节\n\nMessage from: assistant\n如果进一步拆分环节，那么一道牛肉炖土豆的背后，需要更多的人员和环节的努力，包括：\n\n1. **土豆种植环节**：\n\t* **种植户**：\n\t\t+ **土地所有者/租赁者**：负责土地的所有权或租赁。\n\t\t+ **播种者**：负责土豆种子的播种。\n\t\t+ **施肥者**：负责肥料的施用。\n\t\t+ **除草者**：负责除草剂的喷洒。\n\t\t+ **收获者**：负责土豆的收获。\n\t* **肥料生产商**：\n\t\t+ **矿物肥料生产商**：负责矿物肥料的生产。\n\t\t+ **有机肥料生产商**：负责有机肥料的生产。\n\t* **除草剂生产商**：\n\t\t+ **原料供应商**：负责除草剂原料的供应。\n\t\t+ **化工生产商**：负责除草剂的生产。\n\t* **农机操作手**：\n\t\t+ **播种机操作手**：负责播种机的操作。\n\t\t+ **施肥机操作手**：负责施肥机的操作。\n\t\t+ **除草机操作手**：负责除草机的操作。\n\t\t+ **收获机操作手**：负责收获机的操作。\n2. **牛肉运输环节**：\n\t* **车辆制造商**：\n\t\t+ **冶炼厂工人**：负责金属原材料的冶炼。\n\t\t+ **零部件制造商**：负责车辆零部件的制造。\n\t\t+ **总装工人**：负责车辆的总装。\n\t* **路修建设者**：\n\t\t+ **规划设计师**：负责路线的规划和设计。\n\t\t+ **建筑工人**：负责路面的修建。\n\t\t+ **路面养护工人**：负责路面的养护。\n\t* **运输司机**：负责牛肉的运输。\n\t* **维修人员**：\n\t\t+ **维修工**：负责运输车辆的维修。\n\t\t+ **配件供应商**：负责维修配件的供应。\n3. **牛肉屠宰环节**：\n\t* **屠宰场工人**：\n\t\t+ **屠宰工**：负责牛的屠宎。\n\t\t+ **分割工**：负责牛肉的分割。\n\t\t+ **包装工**：负责牛肉的包装。\n\t* **屠宰设备制造商**：\n\t\t+ **机械制造商**：负责屠宰设备的制造。\n\t\t+ **维修人员**：负责屠宰设备的维修。\n4. **水电工人**：\n\t* **水利工程师**：负责水利设施的规划和设计。\n\t* **电力工程师**：负责电力设施的规划和设计。\n\t* **水电工人**：负责水电设施的建设和维护。\n5. **环保工作者**：\n\t* **环保工程师**：负责环保设施的规划和设计。\n\t* **环保工人**：负责环保设施的建设和维护。\n\t* **环保监测人员**：负责环境质量的监测。\n\n通过进一步拆分环节，我们可以看到，一道牛肉炖土豆的背后，需要更多的人员和环节的共同努力，涉及到种植户、肥料生产商、除草剂生产商、农机操作手、土地所有者/租赁者、播种者、施肥者、除草者、收获者、矿物肥料生产商、有机肥料生产商、原料供应商、化工生产商、冶炼厂工人、零部件制造商、总装工人、路修建设者、建筑工人、路面养护工人、规划设计师、运输司机、维修工、配件供应商、屠宰工、分割工、包装工、机械制造商、水利工程师、电力工程师、水电工人、环保工程师、环保工人、环保监测人员等。\n\nMessage from: user\n请继续补充，类似制作牛肉的配料，比如葱姜蒜，花椒，酱油的生产，汽车制造时候需要橡胶等环节，制作过程中需要用到天然气等环节，炉灶的制作等\n\nMessage from: assistant\n如果进一步补充环节，那么一道牛肉炖土豆的背后，还需要考虑以下人员和环节的努力：\n\n1. **配料生产环节**：\n\t* **蔬菜种植户**：\n\t\t+ **葱种植户**：负责葱的种植。\n\t\t+ **姜种植户**：负责姜的种植。\n\t\t+ **蒜种植户**：负责蒜的种植。\n\t* **香辛料生产商**：\n\t\t+ **花椒种植户/采集者**：负责花椒的种植或采集。\n\t\t+ **花椒加工商**：负责花椒的加工。\n\t* **酱油生产商**：\n\t\t+ **大豆种植户**：负责大豆的种植。\n\t\t+ **酿造工**：负责酱油的酿造。\n\t* **调味品生产商**：\n\t\t+ **盐生产商**：负责盐的生产。\n\t\t+ **味精生产商**：负责味精的生产。\n2. **汽车制造环节**：\n\t* **橡胶生产商**：\n\t\t+ **天然橡胶种植户**：负责天然橡胶的种植。\n\t\t+ **合成橡胶生产商**：负责合成橡胶的生产。\n\t* **轮胎制造商**：负责轮胎的制造。\n\t* **汽车零部件制造商**：负责汽车零部件的制造。\n\t* **汽车总装工人**：负责汽车的总装。\n3. **能源供应环节**：\n\t* **天然气生产商**：\n\t\t+ **天然气勘探者**：负责天然气储量的勘探。\n\t\t+ **天然气开采者**：负责天然气的开采。\n\t\t+ **天然气加工商**：负责天然气的加工。\n\t* **电力生产商**：\n\t\t+ **煤炭生产商**：负责煤炭的生产。\n\t\t+ **石油生产商**：负责石油的生产。\n\t\t+ **核电站工人**：负责核电站的运行。\n\t\t+ **风电/光伏发电商**：负责风电/光伏发电的运行。\n4. **炉灶制造环节**：\n\t* **原材料供应商**：\n\t\t+ **钢铁生产商**：负责钢铁的生产。\n\t\t+ **陶瓷生产商**：负责陶瓷的生产。\n\t* **炉灶制造商**：\n\t\t+ **设计师**：负责炉灶的设计。\n\t\t+ **制造工人**：负责炉灶的制造。\n\t* **维修人员**：负责炉灶的维修。\n\n通过进一步补充环节，我们可以看到，一道牛肉炖土豆的背后，还需要考虑配料生产环节、汽车制造环节、能源供应环节、炉灶制造环节等，涉及到蔬菜种植户、香辛料生产商、酱油生产商、调味品生产商、橡胶生产商、轮胎制造商、汽车零部件制造商、汽车总装工人、天然气生产商、电力生产商、原材料供应商、炉灶制造商、维修人员等。\n\nMessage from: user\n还有其他我没有提到的环节吗？\n\n",
"answer_candidate_a": "是的，还有其他一些环节需要考虑，包括：\n\n1. **食品安全监管环节**：\n\t* **食品安全检验员**：负责对食品安全进行检验和监管。\n\t* **食品安全标准制定者**：负责制定食品安全标准。\n2. **知识传播环节**：\n\t* **教育工作者**：负责传授烹饪技能和食品安全知识。\n\t* **媒体工作者**：负责通过媒体传播烹饪技巧和食品安全知识。\n3. **文化传承环节**：\n\t* **传统文化保护者**：负责保护和传承传统的烹饪技艺和食品文化。\n4. **社会福利环节**：\n\t* **慈善机构**：负责帮助社会上的弱势群体，确保他们能够获得足够的食物。\n5. **国际贸易环节**：\n\t* **国际贸易商**：负责国际贸易，包括进口牛肉、土豆、配料等。\n\n通过考虑这些环节，我们可以看到，一道牛肉炖土豆的背后，还需要考虑食品安全监管环节、知识传播环节、文化传承环节、社会福利环节、国际贸易环节等，涉及到食品安全检验员、食品安全标准制定者、教育工作者、媒体工作者、传统文化保护者、慈善机构、国际贸易商等。",
"answer_candidate_b": "是的，还有其他可能没有提到的环节，例如：\n\n1. **包装材料生产环节**：包括塑料、纸、玻璃等材料的生产，用于牛肉、土豆、配料等食品的包装。\n2. **冷链运输环节**：包括冷藏车辆、冷库等设施的运营，用于保证牛肉、土豆等食品在运输过程中的新鲜度。\n3. **食品加工环节**：包括牛肉、土豆等食品的加工、切割、煮熟等过程，用于提高食品的口感和舒适度。\n4. **食品检测环节**：包括食品安全、质量、营养等方面的检测，用于确保食品的安全性和可靠性。\n5. **食品销售环节**：包括超市、餐厅、食品商店等场所的销售，用于将食品送到消费者手中。\n6. **食品消费环节**：包括家庭、餐厅等场所的食品消费，用于满足人们的饮食需求。\n7. **垃圾处理环节**：包括食品废弃物的处理和回收，用于保护环境和资源的可持续利用。\n\n这些环节也需要不同的人员和设备的参与和支持，因此，一道牛肉炖土豆的背后，需要更多的人员和环节的共同努力。",
"preference": "-1"

Example 4:
"conversation_history": "Message from: user\n請提供一份資管系的論文第二章、文件探討的範例，說明第二章需要呈現的章節及內容。\n\n",
"answer_candidate_a": "第二章：文件探討\n\n**一、章節及內容**\n\n第二章主要目的是介紹並探討與資訊管理相關的文件，包括其定義、特性、分類、格式、管理流程等。以下是第二章應包含的章節及內容範例：\n\n**1. 導言**\n\n- 簡單介紹第二章的目的和結構。\n- 解釋為何文件探討對資訊管理而言至關重要。\n\n**2. 文件的定義與特性**\n\n- 定義文件的概念，並與其他相關術語（如資訊、資料）做比較。\n- 列出文件的特性，例如：\n  - 可傳遞性：文件可以被複製和傳遞。\n  - 可讀性：文件可以被人們讀取和理解。\n  - 可控制性：文件的版本和變更可以被控制和跟蹤。\n  - 可保存性：文件可以被長期保存和存放。\n\n**3. 文件的分類**\n\n- 依照不同的標準，將文件分為不同的類別，例如：\n  - 依據格式：紙本文件、電子文件。\n  - 依據內容：文本文件、圖形文件、音訊文件、視頻文件等。\n  - 依據用途：公文、合同、報告、記錄等。\n  - 依據保密等級：機密文件、-confidential文件、公開文件等。\n\n**4. 文件格式**\n\n- 介紹常見的文件格式，例如：\n  - 紙本文件：A4紙、信封、卡片等。\n  - 電子文件：Word、Excel、PDF、HTML等。\n- 討論不同格式的優缺點，並解釋何時應使用特定格式。\n\n**5. 文件管理流程**\n\n- 介紹文件管理的生命週期，包括文件的創建、傳遞、存放、查找、更新、保存和銷毀等過程。\n- 討論文件管理流程中的控制點，例如文件的版本控制、文件的授權和訪問控制等。\n- 解釋文件管理流程的自動化，例如使用文件管理系統（Document Management System, DMS）或企業內容管理系統（Enterprise Content Management, ECM）來管理文件。\n\n**6. 文件管理的挑戰與解決方案**\n\n- 討論文件管理面臨的挑戰，例如文件的安全性、完整性、可追蹤性等。\n- 提供解決方案，例如加密、數位簽名、版本控制等。\n\n**7. 結論**\n\n- 總結第二章的主要內容。\n- 強調文件管理在資訊管理中的重要性。\n- 為下一章節做過渡。",
"answer_candidate_b": "第二章 文件探討 (Literature Review)\n\n文件探討是論文中重要的一章，它提供相關研究的背景、理論基礎和現有發現，以幫助讀者了解研究問題的現況和重要性。以下是資管系論文第二章的範例，包括章節和內容：\n\n1. 引言\n\n   簡要介紹第二章的目的和結構，並說明文件探討如何支持研究問題和目標。\n\n2. 研究問題相關理論\n\n   在這一節，介紹與研究問題相關的理論和概念。例如，如果研究問題是關於企業資源規劃 (ERP) 系統的實施，可以介紹 ERP 系統的定義、功能、優點和挑戰。此外，還可以介紹相關的理論，如技術接受模型 (TAM)、統一理論接受和使用技術 (UTAUT) 或組織變革理論。\n\n3. 現有研究的審查\n\n   在這一節，審查現有研究，並按照主題或方法分類。例如，可以按照 ERP 系統實施的不同階段（如規劃、實施、運行和評估）來分類現有研究。對於每個類別，應該提供現有研究的摘要、發現和結論。此外，還應該指出現有研究的局限性和研究空白，以説明研究問題的必要性。\n\n4. 研究問題相關的實證研究\n\n   在這一節，介紹與研究問題相關的實證研究。例如，如果研究問題是關於 ERP 系統實施的成功因素，可以介紹現有研究中發現的成功因素，如領導力、組織文化、訓練和支持。此外，還應該指出這些研究的局限性和研究空白，以説明研究問題的必要性。\n\n5. 研究問題相關的案例研究\n\n   在這一節，介紹與研究問題相關的案例研究。例如，如果研究問題是關於 ERP 系統實施的挑戰，可以介紹現有案例研究中發現的挑戰，如抵抗變革、缺乏資源和技術問題。此外，還應該指出這些案例研究的局限性和研究空白，以説明研究問題的必要性。\n\n6. 總結\n\n   在這一節，總結第二章的要點，並說明文件探討如何支持研究問題和目標。此外，還應該指出研究問題的必要性和研究空白，以引導讀者進入下一章。\n\n在撰寫第二章時，應該確保文件探討與研究問題相關，並提供足夠的背景和理論基礎。此外，還應該確保文件探討是最新的，並引用相關的學術文獻。",
"preference": "+2"
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
