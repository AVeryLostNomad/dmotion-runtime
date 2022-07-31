using System.Runtime.CompilerServices;
using Latios.Kinemation;
using Latios.Kinemation.Systems;
using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Entities;
using Unity.Jobs;
using Unity.Jobs.LowLevel.Unsafe;
using Unity.Mathematics;
using Unity.Transforms;

namespace DMotion
{
    [UpdateInGroup(typeof(TransformSystemGroup))]
    [UpdateBefore(typeof(TRSToLocalToParentSystem))]
    [UpdateBefore(typeof(TRSToLocalToWorldSystem))]
    public partial class AnimationStateMachineSystem : SystemBase
    {
        private EntityQuery _optimizedQuery;
        private EntityQuery _nonOptimizedQuery;
        private EntityQuery _sampleRootDeltasQuery;

        protected override void OnCreate()
        {
            base.OnCreate();
            _optimizedQuery = GetEntityQuery(
                typeof(OptimizedBoneToRoot),
                typeof(ClipSampler),
                typeof(OptimizedSkeletonHierarchyBlobReference)
            );
            _nonOptimizedQuery = GetEntityQuery(
                ComponentType.ReadOnly<BoneIndex>(),
                ComponentType.ReadWrite<BoneOwningSkeletonReference>(),
                ComponentType.ReadWrite<Translation>(),
                ComponentType.ReadWrite<Rotation>(),
                ComponentType.ReadWrite<NonUniformScale>()
            );
            _sampleRootDeltasQuery = GetEntityQuery(
                typeof(RootDeltaTranslation),
                typeof(RootDeltaRotation),
                typeof(ClipSampler)
            );
        }

        protected override void OnUpdate()
        {
            var updateFmsHandle = new UpdateStateMachineJob()
            {
                DeltaTime = Time.DeltaTime,
            }.ScheduleParallel();
            
            //Sample bones (those only depend on updateFmsHandle)
            var sampleOptimizedHandle = new SampleOptimizedBonesJob()
            {
                samplersHandle = GetBufferTypeHandle<ClipSampler>(true),
                boneToRootBufferHandle = GetBufferTypeHandle<OptimizedBoneToRoot>(),
                hierarchyReferenceHandle = GetComponentTypeHandle<OptimizedSkeletonHierarchyBlobReference>(true)
            }.ScheduleParallel(_optimizedQuery, updateFmsHandle);
            var sampleNonOptimizedHandle = new SampleNonOptimizedBones()
            {
                clipSamplerFromEntity = GetBufferFromEntity<ClipSampler>(),
                translationTypeHandle = GetComponentTypeHandle<Translation>(),
                rotationTypeHandle = GetComponentTypeHandle<Rotation>(),
                scaleTypeHandle = GetComponentTypeHandle<NonUniformScale>(),
                skeletonRefHandle = GetComponentTypeHandle<BoneOwningSkeletonReference>(),
                boneIndexHandle = GetComponentTypeHandle<BoneIndex>(),
            }.ScheduleParallel(_nonOptimizedQuery, JobHandle.CombineDependencies(updateFmsHandle, sampleOptimizedHandle));
            
            var sampleRootDeltasHandle = new SampleRootDeltasJob()
            {
                samplersHandle = GetBufferTypeHandle<ClipSampler>(),
                rootDeltaRotationHandle = GetComponentTypeHandle<RootDeltaRotation>(),
                rootDeltaTranslationHandle = GetComponentTypeHandle<RootDeltaTranslation>()
            }.ScheduleParallel(_sampleRootDeltasQuery, JobHandle.CombineDependencies(updateFmsHandle, sampleNonOptimizedHandle));
            
            var applyRootMotionHandle = new ApplyRootMotionToEntityJob()
            {
            }.ScheduleParallel(sampleRootDeltasHandle);
            
            var transferRootMotionHandle = new TransferRootMotionJob()
            {
                CfeDeltaPosition = GetComponentDataFromEntity<RootDeltaTranslation>(true),
                CfeDeltaRotation = GetComponentDataFromEntity<RootDeltaRotation>(true),
            }.ScheduleParallel(sampleRootDeltasHandle);
            //end sample bones
            
            Dependency = JobHandle.CombineDependencies(sampleOptimizedHandle, sampleNonOptimizedHandle, transferRootMotionHandle);
            Dependency = JobHandle.CombineDependencies(Dependency, applyRootMotionHandle);
        }

        [BurstCompile]
        private struct SampleRootDeltasJob : IJobEntityBatch
        {
            [NativeDisableContainerSafetyRestriction] public BufferTypeHandle<ClipSampler> samplersHandle;
            public ComponentTypeHandle<RootDeltaTranslation> rootDeltaTranslationHandle;
            public ComponentTypeHandle<RootDeltaRotation> rootDeltaRotationHandle;

            public void Execute(ArchetypeChunk chunk, int batchIndex)
            {
                var samplersArray = chunk.GetBufferAccessor(samplersHandle);
                var rootDeltas = chunk.GetNativeArray(rootDeltaTranslationHandle);
                var rootRotations = chunk.GetNativeArray(rootDeltaRotationHandle);

                for (var index = 0; index < chunk.Count; index += 1)
                {
                    var samplers = samplersArray[index];
                    var rootDeltaTranslation = rootDeltas[index];
                    var rootDeltaRotation = rootRotations[index];
                    
                    rootDeltaTranslation.Value = 0;
                    rootDeltaRotation.Value = quaternion.identity;
                    if (samplers.Length > 0 && TryGetFirstSamplerIndex(samplers, out var startIndex))
                    {
                        var firstSampler = samplers[startIndex];
                        var root = ClipSamplingUtils.SampleWeightedFirstIndex(
                            0, ref firstSampler.Clip,
                            firstSampler.NormalizedTime,
                            firstSampler.Weight);

                        var previousRoot = ClipSamplingUtils.SampleWeightedFirstIndex(
                            0, ref firstSampler.Clip,
                            firstSampler.PreviousNormalizedTime,
                            firstSampler.Weight);

                        for (var i = startIndex + 1; i < samplers.Length; i++)
                        {
                            var sampler = samplers[i];
                            if (ShouldIncludeSampler(sampler))
                            {
                                ClipSamplingUtils.SampleWeightedNIndex(
                                    ref root, 0, ref sampler.Clip,
                                    sampler.NormalizedTime, sampler.Weight);

                                ClipSamplingUtils.SampleWeightedNIndex(
                                    ref previousRoot, 0, ref sampler.Clip,
                                    sampler.PreviousNormalizedTime, sampler.Weight);
                            }
                        }

                        rootDeltaTranslation.Value = root.translation - previousRoot.translation;
                        rootDeltaRotation.Value = mathex.delta(root.rotation, previousRoot.rotation);
                    }

                    rootDeltas[index] = rootDeltaTranslation;
                    rootRotations[index] = rootDeltaRotation;
                }
            }

            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            private static bool ShouldIncludeSampler(in ClipSampler sampler)
            {
                //Since we're calculating deltas, we need to avoid the loop point (the character would teleport back to the initial root position)
                return !mathex.iszero(sampler.Weight) && sampler.NormalizedTime - sampler.PreviousNormalizedTime > 0;
            }

            private static bool TryGetFirstSamplerIndex(in DynamicBuffer<ClipSampler> samplers, out byte startIndex)
            {
                for (byte i = 0; i < samplers.Length; i++)
                {
                    if (ShouldIncludeSampler(samplers[i]))
                    {
                        startIndex = i;
                        return true;
                    }
                }
                startIndex = 0;
                return false;
            }
        }

        [BurstCompile]
        private struct SampleOptimizedBonesJob : IJobEntityBatch
        {
            public BufferTypeHandle<OptimizedBoneToRoot> boneToRootBufferHandle;
            [ReadOnly] public BufferTypeHandle<ClipSampler> samplersHandle;
            [ReadOnly] public ComponentTypeHandle<OptimizedSkeletonHierarchyBlobReference> hierarchyReferenceHandle;

            public void Execute(ArchetypeChunk batchInChunk, int batchIndex)
            {
                var boneToRoots = batchInChunk.GetBufferAccessor(boneToRootBufferHandle);
                var samplersArray = batchInChunk.GetBufferAccessor(samplersHandle);
                var hierarchies = batchInChunk.GetNativeArray(hierarchyReferenceHandle);

                for (var index = 0; index < batchInChunk.Count; index += 1)
                {
                    var boneToRootBuffer = boneToRoots[index];
                    var samplers = samplersArray[index];
                    var hierarchyRef = hierarchies[index];

                    var blender = new BufferPoseBlender(boneToRootBuffer);
                    var activeSamplerCount = 0;

                    for (byte i = 0; i < samplers.Length; i++)
                    {
                        var sampler = samplers[i];
                        if (!mathex.iszero(sampler.Weight))
                        {
                            activeSamplerCount++;
                            sampler.Clip.SamplePose(ref blender, sampler.Weight, sampler.NormalizedTime);
                        }
                    }
            
                    if (activeSamplerCount > 1)
                    {
                        blender.NormalizeRotations();
                    }
                    blender.ApplyBoneHierarchyAndFinish(hierarchyRef.blob);
                }
            }
        }

        [BurstCompile]
        private struct SampleNonOptimizedBones : IJobEntityBatch
        {
            [NativeDisableContainerSafetyRestriction] public BufferFromEntity<ClipSampler> clipSamplerFromEntity;
            public ComponentTypeHandle<Translation> translationTypeHandle;
            public ComponentTypeHandle<Rotation> rotationTypeHandle;
            public ComponentTypeHandle<NonUniformScale> scaleTypeHandle;
            public ComponentTypeHandle<BoneOwningSkeletonReference> skeletonRefHandle;
            public ComponentTypeHandle<BoneIndex> boneIndexHandle;

            public void Execute(ArchetypeChunk batchInChunk, int batchIndex)
            {
                var translations = batchInChunk.GetNativeArray(translationTypeHandle);
                var rotations = batchInChunk.GetNativeArray(rotationTypeHandle);
                var scales = batchInChunk.GetNativeArray(scaleTypeHandle);
                var skeletonRefs = batchInChunk.GetNativeArray(skeletonRefHandle);
                var boneIndexes = batchInChunk.GetNativeArray(boneIndexHandle);
                
                for (var iter = 0; iter < batchInChunk.Count; iter += 1)
                {
                    var translation = translations[iter];
                    var rotation = rotations[iter];
                    var scale = scales[iter];
                    var skeletonRef = skeletonRefs[iter];
                    var boneIndex = boneIndexes[iter];
                    
                    var samplers = clipSamplerFromEntity[skeletonRef.skeletonRoot];

                    if (samplers.Length > 0 && TryFindFirstActiveSamplerIndex(samplers, out var firstSamplerIndex))
                    {
                        var firstSampler = samplers[firstSamplerIndex];
                        var bone = ClipSamplingUtils.SampleWeightedFirstIndex(
                            boneIndex.index, ref firstSampler.Clip,
                            firstSampler.NormalizedTime,
                            firstSampler.Weight);
                
                        for (var i = firstSamplerIndex + 1; i < samplers.Length; i++)
                        {
                            var sampler = samplers[i];
                            if (!mathex.iszero(sampler.Weight))
                            {
                                ClipSamplingUtils.SampleWeightedNIndex(
                                    ref bone, boneIndex.index, ref sampler.Clip,
                                    sampler.NormalizedTime, sampler.Weight);
                            }
                        }
                
                        translation.Value = bone.translation;
                        rotation.Value = bone.rotation;
                        scale.Value = bone.scale;
                    }

                    translations[iter] = translation;
                    rotations[iter] = rotation;
                    scales[iter] = scale;
                }
            }

            private bool TryFindFirstActiveSamplerIndex(in DynamicBuffer<ClipSampler> samplers, out byte samplerIndex)
            {
                for (byte i = 0; i < samplers.Length; i++)
                {
                    if (!mathex.iszero(samplers[i].Weight))
                    {
                        samplerIndex = i;
                        return true;
                    }
                }

                samplerIndex = 0;
                return false;
            }
        }
    }
}