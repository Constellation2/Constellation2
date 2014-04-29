using Cudafy;
using Cudafy.Host;
using Cudafy.Translator;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Constellation2
{
    class Program
    {
        static void Main(string[] args)
        {
            SimInit();
            var hw=_gpu.CopyToDevice("HallaTralla");
            _gpu.Launch().Dummy();
            char[] hostHW = new char["HallaTralla".Length];
            _gpu.CopyFromDevice(hw, hostHW);
        }
        private static void SimInit()
        {
            Console.WriteLine("Deserializing class");
            CudafyModule km = CudafyModule.TryDeserialize(typeof(Program).Name);
            Console.WriteLine("Got: " + km);
            var tvc = km == null ? false : km.TryVerifyChecksums();
            Console.WriteLine("TVC: " + tvc);

            if (km == null || !tvc)
            {
                Console.WriteLine("Serializing");
                km = CudafyTranslator.Cudafy(typeof(Program));
                km.Serialize();
            }

            Console.WriteLine("Requesting device");
            _gpu = CudafyHost.GetDevice(eGPUType.Cuda);
            if (_gpu == null)
            {
                _gpu = CudafyHost.GetDevice(eGPUType.OpenCL);
                if (_gpu == null)
                {
                    _gpu = CudafyHost.GetDevice(eGPUType.Emulator);
                    if (_gpu == null)
                    {
                        Console.WriteLine("No deivce found!");
                        return;
                    }
                }
                else Console.WriteLine("Got OpenCL Device: " + _gpu.DeviceId);
            }
            else Console.WriteLine("Got CUDA Device: " + _gpu.DeviceId);
            Console.WriteLine("Loading module");
            _gpu.LoadModule(km);
        }
        [Cudafy]
        public static int Gettid(GThread thread)
        {
            int tid = thread.blockIdx.x * thread.blockDim.x + thread.threadIdx.x;
            return tid;
        }
        [Cudafy]
        public static void Dummy(GThread thread, uint[] A, uint[] B)
        {
        }

        public static GPGPU _gpu { get; set; }
    }
}
