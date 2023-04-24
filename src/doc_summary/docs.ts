import { OpenAI } from "langchain/llms/openai";
import { loadSummarizationChain } from "langchain/chains";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import * as fs from "fs";
import * as dotenv from "dotenv";
import path from "path";
import { dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

dotenv.config();
const __dirname = dirname(fileURLToPath(import.meta.url));

const run = async () => {
    // In this example, we use a `MapReduceDocumentsChain` specifically prompted to summarize a set of documents.
    const text = fs.readFileSync(path.join(__dirname, "./resume.txt")).toString();
    
    const model = new OpenAI({
        temperature: 0,
        openAIApiKey: process.env.OPENAI_API_KEY,
    });
    const textSplitter = new RecursiveCharacterTextSplitter({ chunkOverlap:100, chunkSize: 150 });
    const docs = await textSplitter.createDocuments([text]);

    console.log(docs);

    // This convenience function creates a document chain prompted to summarize a set of documents.
    const query = "请用中文总结，请用中文总结，请用中文总结";
    const chain = loadSummarizationChain(model);
    const res = await chain.call({
        input_documents: docs,
        query
    });
    console.log({ res });
};

export default { run };