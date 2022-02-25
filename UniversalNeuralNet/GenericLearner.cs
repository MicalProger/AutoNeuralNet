using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace UniversalNeuralNet
{
    internal class GenericLearner
    {
        public int ChangingLinks;
        public NeuralCore BestNet;
        public int PopSize;
        public int MaxDelta;
        public Dictionary<int, (double score, NeuralCore net)> Population;
        public GenericLearner(int popSize, int maxDelta, NeuralCore startNet, int changing)
        {
            PopSize = popSize;
            MaxDelta = maxDelta;
            Population = new Dictionary<int, (double score, NeuralCore net)>(popSize);
            BestNet = startNet;
            ChangingLinks = changing;
        }

        public void GeneratePopulation()
        {
            Random rand = new Random();
            var baseMtxs = BestNet.links;
            NeuralCore localCore;
            for (int i = 0; i < PopSize; i++)
            {
                localCore = new NeuralCore(new int[] { 3, 1 });
                localCore.SetAllMatrixes(baseMtxs);
                for (int j = 0; j < ChangingLinks; j++)
                {
                    int cLayer = rand.Next(baseMtxs.Length);
                    int aDim = rand.Next(baseMtxs[cLayer].GetLength(0));
                    int bDim = rand.Next(baseMtxs[cLayer].GetLength(1));
                    var delta = rand.NextDouble(-MaxDelta, MaxDelta);
                    localCore.links[cLayer][aDim, bDim] += delta;
                }
            }
        }

        public void ResetBest()
        {
            BestNet = Population.Values.MaxBy(i => i.score).net;
        }
    }
}
