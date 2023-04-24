import * as dotenv from "dotenv";
import { loadQARefineChain } from "langchain/chains";
import { OpenAI } from "langchain/llms/openai";
import { TextLoader } from "langchain/document_loaders/fs/text";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";

import { fileURLToPath } from 'node:url';
import path, { dirname } from 'node:path';

dotenv.config();
const __dirname = dirname(fileURLToPath(import.meta.url));

const run = async () => {
  // Create the models and chain
  // https://js.langchain.com/docs/modules/models/embeddings/integrations
  const embeddings = new OpenAIEmbeddings();
  const model = new OpenAI({ temperature: 0 });
  const chain = loadQARefineChain(model);

  // Load the documents and create the vector store
  const loader = new TextLoader(path.join(__dirname, "./bee.txt"));
  const docs = await loader.loadAndSplit();
  // https://js.langchain.com/docs/modules/indexes/vector_stores/integrations/memory
  const store = await MemoryVectorStore.fromDocuments(docs, embeddings);

  // Select the relevant documents
  const question = "什么蜜蜂没有蛰针，请用中文回答。";
  const relevantDocs = await store.similaritySearch(question);

  // Call the chain
  const res = await chain.call({
    input_documents: relevantDocs,
    question,
  });

  console.log(res);
  /*
  {
    output_text: '\n' +
      '\n' +
      "The president said that Justice Stephen Breyer has dedicated his life to serve this country and thanked him for his service. He also mentioned that Judge Ketanji Brown Jackson will continue Justice Breyer's legacy of excellence, and that the constitutional right affirmed in Roe v. Wade—standing precedent for half a century—is under attack as never before. He emphasized the importance of protecting access to health care, preserving a woman's right to choose, and advancing maternal health care in America. He also expressed his support for the LGBTQ+ community, and his commitment to protecting their rights, including offering a Unity Agenda for the Nation to beat the opioid epidemic, increase funding for prevention, treatment, harm reduction, and recovery, and strengthen the Violence Against Women Act."
  }
  */
};

export default { run };