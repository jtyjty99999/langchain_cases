import * as dotenv from "dotenv";
import { OpenAI } from "langchain";
import path  from "path";
// https://js.langchain.com/docs/modules/chains/index_related_chains/document_qa
import { loadQAStuffChain, loadQAMapReduceChain } from "langchain/chains";
// https://js.langchain.com/docs/modules/indexes/document_loaders/examples/file_loaders/unstructured
import { TextLoader } from "langchain/document_loaders";
import { fileURLToPath } from 'node:url';
import { dirname } from 'node:path';

dotenv.config();
const __dirname = dirname(fileURLToPath(import.meta.url));

const openaichat = new OpenAI({
  modelName: "gpt-3.5-turbo",
  openAIApiKey: process.env.OPENAI_API_KEY,
});

const loader = new TextLoader(path.join(__dirname, "./work.txt"));
const docs = await loader.load();

const chain = loadQAStuffChain(openaichat);
const query = "3月份，主要的工作重点是什么，请用中文回答";
const query2 = "总结下3月份的工作, 请用中文回答";


const result = await chain.call({
  "input_documents": docs,
  "question": query2
});
console.log(result);
